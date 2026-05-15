# tests/test_metacog.py
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config.settings as settings

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
settings.DB_PATH = _tmp.name
_tmp.close()

import db
db.DB_PATH = settings.DB_PATH

from db.answers import save_answer
from db.documents import upsert_document
from db.metacog import ensure_profile, get_history
from db.schema import initialize_schema
from db.sessions import start_session
from db.subjects import get_subject, get_subject_history_by_subject
from metacog.gauges import initialize_session_gauges, make_gauges, update_gauges_from_evaluation, update_profile_gauges_from_session
from metacog.profile import compute_alpha, update_profile, update_retention_from_quiz
from metacog.reflection import fallback_meta_cognition_analysis, normalize_meta_cognition_questions
from metacog.session import SessionManager
from metacog.signals import compute_session_score


def setup():
    initialize_schema()


def test_alpha_formula():
    assert compute_alpha(0) == 1.0
    assert round(compute_alpha(5), 2) == 0.5
    assert compute_alpha(500) == 0.05


def test_default_profile():
    setup()
    profile = ensure_profile(1)
    assert profile["context_comprehension"] == 50.0
    assert profile["attention"] == 50.0
    assert profile["meta_cognition"] == 50.0


def test_initialize_session_gauges_inherits_all_profile_values():
    values = initialize_session_gauges({
        "attention": 20,
        "context_comprehension": 80,
        "creativity": 50,
        "retention": 40,
        "curiosity": 60,
        "meta_cognition": 70,
    })

    assert values["attention"] == 16.0
    assert values["context_comprehension"] == 64.0
    assert values["meta_cognition"] == 56.0


def test_update_profile_gauges_from_session_is_slow_and_clamped():
    values = update_profile_gauges_from_session(
        {"attention": 50, "curiosity": 50, "meta_cognition": 50},
        {"attention": 100, "curiosity": 120, "meta_cognition": 0},
    )

    assert values["attention"] == 55.0
    assert values["curiosity"] == 55.0
    assert values["meta_cognition"] == 45.0


def test_compute_session_score():
    answers = [
        {
            "verdict": "correct",
            "response_time_ms": 1500,
            "metacog_signals": {
                "context_comprehension": 0.5,
                "creativity": 0.0,
                "attention": 0.2,
                "retention": 0.0,
                "curiosity": 0.0,
                "meta_cognition": 0.0,
            },
        },
        {
            "verdict": "partial",
            "response_time_ms": 5000,
            "metacog_signals": {
                "context_comprehension": 0.1,
                "creativity": 0.0,
                "attention": -0.1,
                "retention": 0.0,
                "curiosity": 0.0,
                "meta_cognition": 0.0,
            },
        },
    ]

    score = compute_session_score(answers, {"attention": 80})

    assert 0 <= score["context_comprehension"] <= 100
    assert score["attention"] > 50


def test_update_profile_records_history():
    setup()
    profile = update_profile(1, {
        "attention": 70,
        "context_comprehension": 80,
        "creativity": 50,
        "retention": 50,
        "curiosity": 50,
        "meta_cognition": 50,
    }, session_id=None)

    assert profile["context_comprehension"] == 53.0
    assert profile["sessions_count"] >= 1
    assert len(get_history(1)) >= 6


def test_quiz_answer_updates_retention_profile():
    setup()
    before = ensure_profile(1)["retention"]

    profile = update_retention_from_quiz(1, "correct", session_id=None)

    assert profile["retention"] > before
    retention_history = [row for row in get_history(1, "retention") if row["session_score"] == 100.0]
    assert retention_history


def test_gauge_attention_decreases_on_long_incorrect_answer():
    gauges = make_gauges({
        "attention": 50,
        "context_comprehension": 50,
        "creativity": 50,
        "retention": 50,
        "curiosity": 50,
        "meta_cognition": 50,
    })
    before = gauges["attention"].value
    values = update_gauges_from_evaluation(
        gauges,
        {
            "verdict": "incorrect",
            "metacog_signals": {
                "context_comprehension": 0,
                "creativity": 0,
                "attention": -0.5,
                "retention": 0,
                "curiosity": 0,
                "meta_cognition": 2,
            },
        },
        response_time_ms=20000,
        consecutive_incorrect=2,
    )

    assert values["attention"] < before
    assert values["meta_cognition"] == 40.0


def test_curiosity_and_creativity_signals_increase_gauges():
    gauges = make_gauges({
        "attention": 50,
        "context_comprehension": 50,
        "creativity": 50,
        "retention": 50,
        "curiosity": 50,
        "meta_cognition": 50,
    })
    before_curiosity = gauges["curiosity"].value
    before_creativity = gauges["creativity"].value

    values = update_gauges_from_evaluation(
        gauges,
        {
            "verdict": "partial",
            "metacog_signals": {
                "context_comprehension": 0,
                "creativity": 0,
                "attention": 0,
                "retention": 0,
                "curiosity": 0,
                "meta_cognition": 0,
            },
            "curiosity_signals": {"asked_follow_up_question": True},
            "creativity_signals": {"makes_connections": True, "depth_of_reflection": 0.8},
        },
    )

    assert values["curiosity"] > before_curiosity
    assert values["creativity"] > before_creativity


def test_meta_cognition_questions_are_exactly_three_and_unique():
    questions = normalize_meta_cognition_questions([
        "Quel point reste fragile ?",
        "Quel point reste fragile ?",
    ])

    assert len(questions) == 3
    assert len(set(questions)) == 3


def test_meta_cognition_analysis_positive_and_negative():
    positive = fallback_meta_cognition_analysis(
        ["Q1", "Q2", "Q3"],
        [
            "J'ai été bloqué par la notation, puis j'ai relu et reformulé avec un exemple.",
            "Je comprends mieux le lien principal mais le dernier passage reste fragile.",
            "Ma stratégie a été de comparer avec le chapitre précédent.",
        ],
    )
    negative = fallback_meta_cognition_analysis(["Q1", "Q2", "Q3"], ["", "", ""])

    assert positive["score_delta"] > 0
    assert negative["score_delta"] < 0


def test_session_answers_feed_score():
    setup()
    doc_id = upsert_document("/tmp/metacog.pdf", "metacog.pdf", 1, "pymupdf", False)
    session_id = start_session(doc_id)
    save_answer(
        question_id=None,
        user_id=1,
        answer_text="réponse",
        verdict="correct",
        response_time_ms=1000,
        metacog_signals={
            "attention": 1,
            "context_comprehension": 1,
            "creativity": 0,
            "retention": 0,
            "curiosity": 0,
            "meta_cognition": 0,
        },
        session_id=session_id,
    )
    from db.answers import get_answers_for_session

    score = compute_session_score(get_answers_for_session(session_id), None)
    assert score["context_comprehension"] > 70


def test_subject_gauge_updates_during_session_and_persists():
    setup()
    doc_id = upsert_document(
        "/tmp/math-subject.pdf",
        "math-subject.pdf",
        1,
        "pymupdf",
        False,
        subject="mathématiques",
    )
    manager = SessionManager(doc_id, subject="mathématiques")
    before = float(get_subject(1, "mathématiques")["level"])

    values = manager.update_from_evaluation({
        "verdict": "correct",
        "metacog_signals": {
            "context_comprehension": 1.0,
            "retention": 0.5,
        },
    })

    after = float(get_subject(1, "mathématiques")["level"])
    assert after > before
    assert values["subject"] == after
    history = get_subject_history_by_subject(1)
    assert history["mathématiques"][-1][0] == manager.session_id
    assert history["mathématiques"][-1][1] == after

    other_doc_id = upsert_document(
        "/tmp/other-math-subject.pdf",
        "other-math-subject.pdf",
        1,
        "pymupdf",
        False,
        subject="mathématiques",
    )
    next_manager = SessionManager(other_doc_id, subject="mathématiques")
    assert next_manager.subject_level == after
