from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from config.settings import (
    LLM_LAYOUT_TIMEOUT,
    OLLAMA_MODEL,
    PDF_LLM_MAX_ORDER_ANCHORS,
    PDF_LLM_MAX_ORDER_BLOCKS,
)
from db.llm_pdf_cache import get_llm_pdf_cache, save_llm_pdf_cache
from document.global_index import hash_pdf
from document.layout.llm_geometry_validator import validate_reading_order
from document.layout.page_reading_plan import PageReadingPlan
from document.models import DocumentBlock
from llm.pdf_assistant_queue import (
    PDF_LLM_PRIORITIES,
    get_pdf_llm_queue,
    llm_cache_enabled,
    validate_reading_order_response,
)

logger = logging.getLogger("Document.layout.llm_order")


def llm_order_with_cache(
    *,
    pdf_path: str,
    page_number: int,
    blocks: list[DocumentBlock],
    plan: PageReadingPlan,
    prev_page_tail: list[str] | None = None,
    model: str | None = None,
    priority: str = "layout_visible",
    generation: int | None = None,
) -> list[str]:
    fallback = list(plan.reading_order_ids)
    if not fallback:
        return []
    if not _llm_order_budget_allows(fallback, plan):
        logger.info(
            "[LLM_ORDER] ignoré page=%s blocs=%s anchors=%s budget=%s/%s",
            page_number,
            len(fallback),
            len(plan.visual_anchors),
            PDF_LLM_MAX_ORDER_BLOCKS,
            PDF_LLM_MAX_ORDER_ANCHORS,
        )
        return fallback

    summaries = [_block_summary(block) for block in blocks if block.id in set(fallback)]
    input_payload = {
        "pdf_hash": _safe_pdf_hash(pdf_path),
        "page": page_number,
        "previous_page_tail": list(prev_page_tail or [])[-4:],
        "blocks": summaries,
    }
    input_hash = _hash_json(input_payload)
    cache_key = _hash_json(
        {
            "task": "layout_order",
            "pdf_hash": input_payload["pdf_hash"],
            "page": page_number,
            "input_hash": input_hash,
        }
    )

    if llm_cache_enabled():
        try:
            cached = get_llm_pdf_cache(cache_key, "layout_order")
        except Exception as exc:
            logger.debug("[LLM_ORDER] cache indisponible: %s", exc)
            cached = None
        cached_order = validate_reading_order_response((cached or {}).get("output", {}).get("reading_order"), fallback)
        if cached_order is not None:
            logger.info("[LLM_ORDER] cache hit page=%s", page_number)
            return cached_order

    def _call() -> list[str]:
        return validate_reading_order(
            fallback,
            summaries,
            page_image_path=None,
            previous_page_ids=prev_page_tail or [],
            model=model or OLLAMA_MODEL,
        )

    started_order = get_pdf_llm_queue().run_sync(
        priority,
        _call,
        priority=PDF_LLM_PRIORITIES.get(priority, 0),
        timeout=LLM_LAYOUT_TIMEOUT,
        generation=generation,
    )
    candidate = validate_reading_order_response(started_order, fallback)
    if candidate is None:
        logger.info("[LLM_ORDER] fallback déterministe page=%s", page_number)
        return fallback

    if llm_cache_enabled():
        try:
            save_llm_pdf_cache(
                cache_key=cache_key,
                task_type="layout_order",
                input_hash=input_hash,
                output={"reading_order": candidate},
                confidence=1.0,
                model=model or OLLAMA_MODEL,
            )
        except Exception as exc:
            logger.debug("[LLM_ORDER] cache écriture indisponible: %s", exc)
    logger.info("[LLM_ORDER] ordre validé page=%s blocs=%s", page_number, len(candidate))
    return candidate


def _block_summary(block: DocumentBlock) -> dict[str, Any]:
    text = (block.text or block.latex or block.caption or "").strip()
    return {
        "id": block.id,
        "type": block.type,
        "bbox": block.bbox.to_list() if block.bbox else None,
        "text_preview": " ".join(text.split())[:120],
    }


def _llm_order_budget_allows(reading_order_ids: list[str], plan: PageReadingPlan) -> bool:
    """Keep the geometry validator for small ambiguous pages only.

    Large pages create long prompts, monopolize the single PDF LLM worker, and
    frequently timeout. The deterministic PageReadingPlan remains the fallback.
    """
    if len(reading_order_ids) > int(PDF_LLM_MAX_ORDER_BLOCKS):
        return False
    if len(plan.visual_anchors) > int(PDF_LLM_MAX_ORDER_ANCHORS):
        return False
    return True


def _safe_pdf_hash(pdf_path: str) -> str:
    try:
        return hash_pdf(Path(pdf_path))
    except OSError:
        return hashlib.sha256(str(pdf_path).encode("utf-8", errors="ignore")).hexdigest()


def _hash_json(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
