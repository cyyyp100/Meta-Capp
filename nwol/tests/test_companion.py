# tests/test_companion.py
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

from core.companion import AdaptiveCompanion, _preprocess_paragraph_for_llm
from db import get_connection
from db.answers import get_answers_for_session
from db.documents import upsert_document
from db.schema import initialize_schema
from db.sessions import start_session
from db.user import DEFAULT_USER_ID
from i18n import set_lang
from reader.state import ReaderState


def setup():
    initialize_schema()


def _question_generator(context, on_success, on_error):
    on_success({
        "question_type": "open",
        "question": "Que dit le paragraphe ?",
        "choices": [],
        "expected_answer": "Il présente une idée.",
    })


def test_cycle_correct_calls_on_complete():
    setup()
    doc_id = upsert_document("/tmp/companion.pdf", "companion.pdf", 1, "pymupdf", False)
    session_id = start_session(doc_id)
    state = ReaderState(doc_id=doc_id)
    completed = []

    def evaluator(context, on_success, on_error):
        on_success({
            "verdict": "correct",
            "feedback": "Correct.",
            "completion": "",
            "hint": "",
            "metacog_signals": {
                "context_comprehension": 0.5,
                "creativity": 0.0,
                "attention": 0.1,
                "retention": 0.0,
                "curiosity": 0.0,
            },
            "flashcard": None,
        })

    companion = AdaptiveCompanion(
        state=state,
        question_generator=_question_generator,
        answer_evaluator=evaluator,
    )
    companion.start_paragraph_qa({"text": "Un paragraphe.", "page_number": 1}, session_id, lambda: completed.append(True))
    companion.handle_answer("Il présente une idée.", 1200)

    assert completed == [True]
    assert state.qa_active is False
    assert get_answers_for_session(session_id)[0]["verdict"] == "correct"


def test_incorrect_twice_triggers_rephrasing():
    setup()
    doc_id = upsert_document("/tmp/companion2.pdf", "companion2.pdf", 1, "pymupdf", False)
    session_id = start_session(doc_id)
    state = ReaderState(doc_id=doc_id)
    rephrasings = []

    def evaluator(context, on_success, on_error):
        on_success({
            "verdict": "incorrect",
            "feedback": "Non.",
            "completion": "",
            "hint": "Relis le sujet.",
            "metacog_signals": {
                "context_comprehension": -0.5,
                "creativity": 0.0,
                "attention": -0.2,
                "retention": 0.0,
                "curiosity": 0.0,
            },
            "flashcard": None,
        })

    def rephraser(context, on_success, on_error):
        on_success({
            "rephrasing_angle": "définition",
            "rephrased_paragraph": "Vu autrement.",
            "note": "Regarde le mot clé.",
        })

    companion = AdaptiveCompanion(
        state=state,
        user_id=DEFAULT_USER_ID,
        on_rephrasing=lambda item: rephrasings.append(item),
        question_generator=_question_generator,
        answer_evaluator=evaluator,
        rephrasing_generator=rephraser,
    )
    companion.start_paragraph_qa({"text": "Un paragraphe.", "page_number": 1}, session_id)
    companion.handle_answer("mauvais", 1000)
    companion.handle_answer("encore mauvais", 1100)

    assert rephrasings[0]["rephrased_paragraph"] == "Vu autrement."
    assert len(get_answers_for_session(session_id)) >= 2


def test_question_mask_callback_is_called():
    setup()
    doc_id = upsert_document("/tmp/companion-mask.pdf", "companion-mask.pdf", 1, "pymupdf", False)
    session_id = start_session(doc_id)
    state = ReaderState(doc_id=doc_id)
    masks = []

    def question_generator(context, on_success, on_error):
        on_success({
            "question_type": "open",
            "question": "Quel élément est masqué ?",
            "choices": [],
            "expected_answer": "Une idée.",
            "paragraph_mask": {
                "enabled": True,
                "start_char": 3,
                "end_char": 8,
                "placeholder": "réponse masquée temporairement",
            },
        })

    companion = AdaptiveCompanion(
        state=state,
        on_mask=lambda start, end, placeholder: masks.append((start, end, placeholder)),
        question_generator=question_generator,
    )

    companion.start_paragraph_qa({"text": "Une idée apparaît.", "page_number": 1}, session_id)

    assert masks == [(3, 8, "réponse masquée temporairement")]


def test_existing_question_flag_is_passed_to_generator():
    setup()
    captured = {}

    def question_generator(context, on_success, on_error):
        captured.update(context)
        on_success({
            "question_type": "open",
            "question": "Réponds à la question du texte.",
            "choices": [],
            "expected_answer": "Une réponse.",
        })

    companion = AdaptiveCompanion(question_generator=question_generator)
    companion.start_paragraph_qa({"text": "Pourquoi cette méthode fonctionne-t-elle ?", "page_number": 1}, None)

    assert captured["has_existing_question"] is True


def test_question_context_uses_current_block_page_and_source_id():
    setup()
    captured = {}
    questions = []

    def question_generator(context, on_success, on_error):
        captured.update(context)
        on_success({
            "question_type": "open",
            "question": "Que dit ce bloc ?",
            "choices": [],
            "expected_answer": "Le bloc courant.",
        })

    companion = AdaptiveCompanion(
        on_question=lambda question: questions.append(question),
        question_generator=question_generator,
    )
    block = {
        "type": "paragraph",
        "text": "Le deuxième paragraphe présente l'idée réellement lue.",
        "page_start": 7,
        "metadata": {"block_index": 3},
    }

    companion.start_paragraph_qa({"block": block, "paragraph": block["text"]}, None)

    assert captured["paragraph"].startswith("Le deuxième paragraphe")
    assert captured["source_block_id"].startswith("p7:b3:")
    assert questions[0]["source_block_id"] == captured["source_block_id"]
    assert companion.paragraph.label == "Paragraphe p.7 #4"


def test_stale_question_callback_is_ignored_when_paragraph_changed():
    setup()
    callbacks = []
    questions = []

    def delayed_question_generator(context, on_success, on_error):
        callbacks.append((context["source_block_id"], context["paragraph"], on_success))

    companion = AdaptiveCompanion(
        on_question=lambda question: questions.append(question),
        question_generator=delayed_question_generator,
    )

    companion.start_paragraph_qa({
        "paragraph": "Premier paragraphe de la session.",
        "page_start": 1,
        "source_block_id": "p1:b0:first",
    }, None)
    companion.start_paragraph_qa({
        "paragraph": "Deuxième paragraphe réellement actif.",
        "page_start": 1,
        "source_block_id": "p1:b1:second",
    }, None)

    callbacks[0][2]({
        "question_type": "open",
        "question": "Question du premier paragraphe ?",
        "choices": [],
        "expected_answer": "Premier.",
        "source_block_id": callbacks[0][0],
    })
    callbacks[1][2]({
        "question_type": "open",
        "question": "Question du deuxième paragraphe ?",
        "choices": [],
        "expected_answer": "Deuxième.",
        "source_block_id": callbacks[1][0],
    })

    assert [question["question"] for question in questions] == ["Question du deuxième paragraphe ?"]


def test_llm_source_block_id_mismatch_is_corrected_for_active_paragraph():
    setup()
    questions = []

    def question_generator(context, on_success, on_error):
        on_success({
            "question_type": "open",
            "question": "Question active ?",
            "choices": [],
            "expected_answer": "Active.",
            "source_block_id": "llm-wrong-id",
        })

    companion = AdaptiveCompanion(
        on_question=lambda question: questions.append(question),
        question_generator=question_generator,
    )

    companion.start_paragraph_qa({
        "paragraph": "Paragraphe actif.",
        "page_start": 2,
        "source_block_id": "p2:b4:active",
    }, None)

    assert questions[0]["source_block_id"] == "p2:b4:active"


def test_saved_question_records_source_block_id_and_current_context():
    setup()
    doc_id = upsert_document("/tmp/source-block.pdf", "source-block.pdf", 2, "pymupdf", False)
    session_id = start_session(doc_id)
    state = ReaderState(doc_id=doc_id)
    questions = []

    companion = AdaptiveCompanion(
        state=state,
        on_question=lambda question: questions.append(question),
        question_generator=_question_generator,
    )
    block = {
        "type": "paragraph",
        "text": "Ce paragraphe exact doit être sauvegardé comme contexte.",
        "page_start": 2,
        "metadata": {"block_index": 8},
    }

    companion.start_paragraph_qa({"block": block, "paragraph": block["text"]}, session_id)

    row = get_connection().execute(
        "SELECT scope_label, source_block_id, source_context FROM questions WHERE id=?",
        (questions[0]["id"],),
    ).fetchone()
    assert row["scope_label"] == "Paragraphe p.2 #9"
    assert row["source_block_id"].startswith("p2:b8:")
    assert row["source_context"].startswith("Ce paragraphe exact")


def test_low_attention_gauge_is_passed_and_adds_pause_hint():
    setup()
    captured = {}
    questions = []

    def question_generator(context, on_success, on_error):
        captured.update(context)
        on_success({
            "question_type": "metacognition",
            "question": "Comment vérifies-tu ta compréhension ?",
            "choices": [],
            "expected_answer": "Une stratégie de vérification.",
        })

    companion = AdaptiveCompanion(
        on_question=lambda question: questions.append(question),
        question_generator=question_generator,
    )
    companion.start_paragraph_qa({
        "text": "Un paragraphe assez long pour créer une question adaptative.",
        "page_number": 1,
        "session_gauges": {"attention": 40, "context_comprehension": 50},
    }, None)

    assert captured["session_gauges"]["attention"] == 40
    assert "pause courte" in questions[0]["session_hint"]


def test_follow_up_question_returns_feedback_payload():
    setup()
    feedback = []

    def follow_up_answerer(context, on_success, on_error):
        assert context["user_question"] == "Peux-tu préciser ?"
        on_success({
            "answer": "Le paragraphe précise le rôle de l'hypothèse.",
            "metacog_signals": {
                "context_comprehension": 0.0,
                "creativity": 0.0,
                "attention": 0.0,
                "retention": 0.0,
                "curiosity": 1.0,
                "meta_cognition": 0.0,
            },
            "curiosity_signals": {"asked_follow_up_question": True},
        })

    companion = AdaptiveCompanion(
        on_feedback=lambda item: feedback.append(item),
        question_generator=_question_generator,
        follow_up_answerer=follow_up_answerer,
    )
    companion.start_paragraph_qa({"text": "Le paragraphe expose une hypothèse.", "page_number": 1}, None)
    companion.handle_follow_up_question("Peux-tu préciser ?")

    assert feedback[0]["follow_up_answer"].startswith("Le paragraphe")
    assert feedback[0]["verdict"] == "correct"
    assert feedback[0]["curiosity_signals"]["asked_follow_up_question"] is True


def test_preprocess_paragraph_for_llm_adds_table_and_figure_context():
    context = _preprocess_paragraph_for_llm(
        "La suite uₙ vérifie uₙ ̸= 0.",
        [
            {
                "type": "table",
                "markdown": "| n | u_n |\n| --- | --- |\n| 1 | 2 |",
                "metadata": {"rows": 2, "columns": 2},
            },
            {"type": "figure", "caption": "Figure 1 : Courbe de convergence"},
        ],
    )

    assert r"u_n \neq 0" in context
    assert "[Tableau 2×2 lignes×colonnes]" in context
    assert "| n | u_n |" in context
    assert '[Figure sur cette page : "Figure 1 : Courbe de convergence"]' in context


def test_preprocess_paragraph_for_llm_uses_english_context_labels():
    set_lang("en")
    try:
        context = _preprocess_paragraph_for_llm(
            "The module uses point prompts.",
            [
                {
                    "type": "table",
                    "markdown": "| n | value |\n| --- | --- |\n| 1 | 2 |",
                    "metadata": {"rows": 2, "columns": 2},
                },
                {"type": "figure", "caption": "Figure 1. Process overview"},
            ],
        )
    finally:
        set_lang("fr")

    assert "Adjacent context:" in context
    assert "[Table 2×2 rows×columns]" in context
    assert '[Figure on this page: "Figure 1. Process overview"]' in context
    assert "Contexte adjacent" not in context
