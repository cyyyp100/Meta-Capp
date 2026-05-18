# db/pages_cache.py — CRUD table pages_cache
import json
import logging
from db import get_connection

logger = logging.getLogger("DB.pages_cache")

# Bump this whenever PDF block generation changes. The engine key includes this
# value, so old SQLite page caches are ignored without deleting user data.
EXTRACTION_CACHE_VERSION = "pdf_pipeline_v38"


def _cache_engine_key(engine: str) -> str:
    return f"{engine}:{EXTRACTION_CACHE_VERSION}"


def get_cached_page(doc_id: int, page_number: int, engine: str) -> list | None:
    payload = get_cached_page_payload(doc_id, page_number, engine)
    if payload is None:
        return None
    return payload["blocks"]


def get_cached_page_payload(doc_id: int, page_number: int, engine: str) -> dict | None:
    cache_engine = _cache_engine_key(engine)
    conn = get_connection()
    row = conn.execute(
        """SELECT blocks_json, page_plan_json, layout_risk_json, quality_score,
                  warnings_json, enrich_assets, extracted_at
           FROM pages_cache
          WHERE document_id=? AND page_number=? AND engine=?""",
        (doc_id, page_number, cache_engine)
    ).fetchone()
    if row:
        logger.debug(f"Cache hit — doc={doc_id} page={page_number} engine={cache_engine}")
        return {
            "blocks": json.loads(row["blocks_json"]),
            "page_plan": _loads_json(row["page_plan_json"]),
            "layout_risk": _loads_json(row["layout_risk_json"]),
            "quality_score": row["quality_score"],
            "warnings": _loads_json(row["warnings_json"]) or [],
            "enrich_assets": bool(row["enrich_assets"]),
            "extracted_at": row["extracted_at"],
        }
    return None


def cache_page(
    doc_id: int,
    page_number: int,
    engine: str,
    blocks: list,
    *,
    page_plan: dict | None = None,
    layout_risk: dict | None = None,
    quality_score: float | None = None,
    warnings: list[str] | None = None,
    enrich_assets: bool = True,
) -> None:
    cache_engine = _cache_engine_key(engine)
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT INTO pages_cache
               (document_id, page_number, engine, blocks_json, enrich_assets,
                page_plan_json, layout_risk_json, quality_score, warnings_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(document_id, page_number, engine) DO UPDATE SET
                 blocks_json=excluded.blocks_json,
                 enrich_assets=excluded.enrich_assets,
                 page_plan_json=excluded.page_plan_json,
                 layout_risk_json=excluded.layout_risk_json,
                 quality_score=excluded.quality_score,
                 warnings_json=excluded.warnings_json,
                 extracted_at=datetime('now')""",
            (
                doc_id,
                page_number,
                cache_engine,
                json.dumps(blocks, ensure_ascii=False),
                int(bool(enrich_assets)),
                json.dumps(page_plan, ensure_ascii=False) if page_plan is not None else None,
                json.dumps(layout_risk, ensure_ascii=False) if layout_risk is not None else None,
                quality_score,
                json.dumps(warnings or [], ensure_ascii=False),
            )
        )
    logger.debug(f"Cache écrit — doc={doc_id} page={page_number} engine={cache_engine}")


def clear_cached_pages(doc_id: int | None = None, engine: str | None = None) -> int:
    conn = get_connection()
    clauses: list[str] = []
    params: list = []
    if doc_id is not None:
        clauses.append("document_id=?")
        params.append(doc_id)
    if engine:
        clauses.append("engine=?")
        params.append(_cache_engine_key(engine))

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with conn:
        cursor = conn.execute(f"DELETE FROM pages_cache{where}", tuple(params))
    removed = int(cursor.rowcount or 0)
    logger.debug("Cache pages supprimé — doc=%s engine=%s count=%s", doc_id, engine, removed)
    return removed


def count_cached_pages(doc_id: int, engine: str) -> int:
    cache_engine = _cache_engine_key(engine)
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as n FROM pages_cache WHERE document_id=? AND engine=?",
        (doc_id, cache_engine)
    ).fetchone()
    return row["n"] if row else 0


def _loads_json(raw: str | None):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
