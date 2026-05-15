from __future__ import annotations

import json
import logging
from typing import Any

from db import get_connection

logger = logging.getLogger("DB.document_index")


def save_document_index(
    *,
    doc_id: int,
    pdf_hash: str,
    opendataloader_status: str,
    detected_document_type: str | None = None,
    chapters: list[dict[str, Any]] | None = None,
    global_assets: dict[str, Any] | list[dict[str, Any]] | None = None,
    backend_report: dict[str, Any] | None = None,
) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT INTO document_index
               (doc_id, pdf_hash, opendataloader_status, detected_document_type,
                chapters_json, global_assets_json, backend_report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 pdf_hash=excluded.pdf_hash,
                 opendataloader_status=excluded.opendataloader_status,
                 detected_document_type=excluded.detected_document_type,
                 chapters_json=excluded.chapters_json,
                 global_assets_json=excluded.global_assets_json,
                 backend_report_json=excluded.backend_report_json,
                 updated_at=datetime('now')""",
            (
                doc_id,
                pdf_hash,
                opendataloader_status,
                detected_document_type,
                json.dumps(chapters or [], ensure_ascii=False),
                json.dumps(global_assets or {}, ensure_ascii=False),
                json.dumps(backend_report or {}, ensure_ascii=False),
            ),
        )
    logger.info(
        "[ODL_INDEX] index sauvegardé doc=%s status=%s type=%s",
        doc_id,
        opendataloader_status,
        detected_document_type,
    )


def get_document_index(doc_id: int) -> dict[str, Any] | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM document_index WHERE doc_id=?", (doc_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["chapters"] = _loads_json(data.pop("chapters_json", None)) or []
    data["global_assets"] = _loads_json(data.pop("global_assets_json", None)) or {}
    data["backend_report"] = _loads_json(data.pop("backend_report_json", None)) or {}
    return data


def _loads_json(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
