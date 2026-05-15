# tests/test_db.py
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Rediriger la DB vers un fichier temporaire pour les tests
import config.settings as settings
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
settings.DB_PATH = _tmp.name
_tmp.close()

from db.schema import initialize_schema
from db import get_connection
from db.documents import upsert_document, get_document_by_path, update_last_page
from db.pages_cache import cache_page, get_cached_page, count_cached_pages
from db.chapters import save_chapters, get_chapters
from db.questions import save_questions, get_questions_for_scope
from db.questions import save_question
from db.answers import save_answer
from db.quiz_questions import get_quiz_questions
from db.flashcards import get_flashcard, save_flashcard
from db.user import DEFAULT_USER_ID, get_user_speed, save_user_speed


def setup():
    initialize_schema()


def test_upsert_and_get_document():
    setup()
    doc_id = upsert_document("/tmp/test.pdf", "test.pdf", 42, "pymupdf", False)
    assert doc_id > 0
    doc = get_document_by_path("/tmp/test.pdf")
    assert doc is not None
    assert doc["filename"] == "test.pdf"
    assert doc["page_count"] == 42


def test_upsert_existing_document_returns_document_id_after_other_inserts():
    setup()
    doc_id = upsert_document("/tmp/reopen.pdf", "reopen.pdf", 12, "pymupdf", False)
    other_doc_id = upsert_document("/tmp/other.pdf", "other.pdf", 8, "pymupdf", True)
    save_chapters(
        other_doc_id,
        [
            {"title": "Intro", "page_start": 1, "page_end": 4, "toc_level": 1},
            {"title": "Suite", "page_start": 5, "page_end": 8, "toc_level": 1},
        ],
    )

    reopened_id = upsert_document("/tmp/reopen.pdf", "reopen.pdf", 13, "pymupdf", True)

    assert reopened_id == doc_id
    doc = get_document_by_path("/tmp/reopen.pdf")
    assert doc["page_count"] == 13
    assert doc["has_toc"] == 1


def test_update_last_page():
    setup()
    doc_id = upsert_document("/tmp/test2.pdf", "test2.pdf", 10, "pymupdf", True)
    update_last_page(doc_id, 7)
    doc = get_document_by_path("/tmp/test2.pdf")
    assert doc["last_page"] == 7


def test_user_speed_persists():
    setup()
    save_user_speed(DEFAULT_USER_ID, 123)

    assert get_user_speed(DEFAULT_USER_ID) == 123


def test_pages_cache():
    setup()
    doc_id = upsert_document("/tmp/test3.pdf", "test3.pdf", 5, "pymupdf", False)
    blocks = [{"type": "paragraph", "text": "Hello"}]
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pages_cache (document_id, page_number, engine, blocks_json) VALUES (?, ?, ?, ?)",
            (doc_id, 2, "pymupdf", '[{"type":"paragraph","text":"old"}]'),
        )

    assert get_cached_page(doc_id, 2, "pymupdf") is None

    cache_page(doc_id, 2, "pymupdf", blocks)
    result = get_cached_page(doc_id, 2, "pymupdf")
    assert result is not None
    assert result[0]["text"] == "Hello"
    assert count_cached_pages(doc_id, "pymupdf") == 1


def test_chapters():
    setup()
    doc_id = upsert_document("/tmp/test4.pdf", "test4.pdf", 20, "pymupdf", True)
    chapters = [
        {"title": "Chap 1", "page_start": 1, "page_end": 10, "toc_level": 1},
        {"title": "Chap 2", "page_start": 11, "page_end": 20, "toc_level": 1},
    ]
    save_chapters(doc_id, chapters)
    result = get_chapters(doc_id)
    assert len(result) == 2
    assert result[0]["title"] == "Chap 1"


def test_questions():
    setup()
    doc_id = upsert_document("/tmp/test5.pdf", "test5.pdf", 15, "pymupdf", False)
    qs = [
        {"question": "Qu'est-ce que X ?", "answer": "X est Y."},
        {"question": "Pourquoi Z ?", "answer": "Parce que W."},
    ]
    save_questions(doc_id, "chapter", "Chap 1", 1, 10, qs)
    result = get_questions_for_scope(doc_id, 1, 10)
    assert len(result) == 2
    assert result[0]["question"] == "Qu'est-ce que X ?"


def test_quiz_questions_include_reading_course_context():
    setup()
    doc_id = upsert_document(
        "/tmp/quiz-context.pdf",
        "quiz-context.pdf",
        3,
        "pymupdf",
        False,
        subject="informatique",
    )
    qid = save_question(
        doc_id,
        "paragraph",
        "Paragraphe p.2",
        2,
        2,
        {
            "question_type": "comprehension",
            "question": "Quelle méthode doit être utilisée pour obtenir les prédictions ?",
            "expected_answer": "La méthode predict.",
            "source_context": "Le cours explique que la méthode predict produit les prédictions sur les sujets de test.",
        },
    )
    save_answer(qid, 1, "Je ne sais pas.", verdict="incorrect")

    questions = get_quiz_questions(1, n=1, subject="informatique")

    assert questions[0]["source"] == "reading"
    assert questions[0]["category"] == "informatique"
    assert "méthode predict" in questions[0]["source_context"]
    assert "Cours : quiz-context.pdf" in questions[0]["course_context"]


def test_flashcard_without_tags_gets_fallback_tags():
    setup()
    card_id = save_flashcard(
        user_id=1,
        question_id=None,
        front="Définition de la convergence uniforme",
        back="Une suite de fonctions converge uniformément si le supremum tend vers zéro.",
        tags=[],
        source="manual",
    )

    card = get_flashcard(card_id)
    assert len(card["tags"]) >= 1
    assert "important" not in card["tags"]


def test_flashcard_assets_are_stored_in_database(tmp_path):
    setup()
    asset_path = tmp_path / "context.png"
    asset_path.write_bytes(b"fake-png")

    card_id = save_flashcard(
        user_id=1,
        question_id=None,
        front="Que montre le schema ?",
        back="Le schema montre une relation importante.",
        tags=["schema"],
        source="auto",
        asset_paths=[str(asset_path)],
    )

    asset_path.unlink()
    card = get_flashcard(card_id)

    assert len(card["assets"]) == 1
    assert card["assets"][0]["filename"] == "context.png"
    assert card["assets"][0]["data_base64"]
