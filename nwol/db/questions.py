# db/questions.py — CRUD table questions
import json
import logging
from db import get_connection

logger = logging.getLogger("DB.questions")


def save_question(
    doc_id: int,
    scope_type: str,
    scope_label: str,
    page_start: int | None,
    page_end: int | None,
    question: dict,
    llm_model: str | None = None,
    session_id: int | None = None,
    chapter_id: int | None = None,
) -> int:
    choices = question.get("choices") or []
    choices_json = json.dumps(choices, ensure_ascii=False) if choices else None
    source_context = (
        question.get("source_context")
        or question.get("course_context")
        or question.get("context")
        or ""
    )
    source_block_id = str(question.get("source_block_id") or "").strip() or None
    conn = get_connection()
    with conn:
        cur = conn.execute(
            """INSERT INTO questions
               (document_id, session_id, chapter_id, scope_type, scope_label,
                page_start, page_end, question_type, question, source_context, source_block_id,
                choices_json, answer, llm_model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                session_id,
                chapter_id,
                scope_type,
                scope_label,
                page_start,
                page_end,
                question.get("question_type"),
                question["question"],
                str(source_context).strip()[:1800] if source_context else None,
                source_block_id,
                choices_json,
                question.get("answer") or question.get("expected_answer", ""),
                question.get("llm_model") or llm_model,
            ),
        )
    logger.info("Question sauvegardée id=%s type=%s scope='%s'", cur.lastrowid, question.get("question_type"), scope_label)
    return int(cur.lastrowid)


def save_questions(doc_id: int, scope_type: str, scope_label: str,
                   page_start: int, page_end: int,
                   questions: list[dict], llm_model: str | None = None) -> list[int]:
    ids = [
        save_question(doc_id, scope_type, scope_label, page_start, page_end, question, llm_model)
        for question in questions
    ]
    logger.info(f"Questions sauvegardées ({len(questions)}) pour scope '{scope_label}'")
    return ids


def get_questions_for_scope(doc_id: int, page_start: int, page_end: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM questions
           WHERE document_id=? AND page_start=? AND page_end=?
           ORDER BY id""",
        (doc_id, page_start, page_end)
    ).fetchall()
    return [dict(r) for r in rows]
