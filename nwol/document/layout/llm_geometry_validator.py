"""LLM-based geometric reading-order validator.

Contract:
  - Input: existing block IDs + compact block JSON + optional low-res page image path
           + previous-page context (list of IDs already read)
  - Output: re-ordered list of the *same* block IDs
  - The LLM may NOT create new IDs, introduce text, or invent image paths.
  - On low-confidence or invalid response → deterministic fallback is returned.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from config.settings import OLLAMA_MODEL, OLLAMA_OPTIONS, OLLAMA_TIMEOUT, OLLAMA_URL, OLLAMA_KEEP_ALIVE

logger = logging.getLogger("Document.layout.llm_geometry")

_VALIDATION_PROMPT_TEMPLATE = """\
You are a geometric reading-order validator. Your ONLY task is to return the block IDs in correct reading order.

Rules:
- Return ONLY a JSON object: {{"reading_order": ["id1", "id2", ...]}}
- Include ONLY IDs from the list below — never invent new ones.
- Do not create text, image paths, or any content.
- Reading order: full-width blocks first (by y position), then left column (top→bottom), then right column (top→bottom).

Previous page last IDs (already read, do not repeat): {prev_ids}

Blocks (JSON):
{blocks_json}
"""


def validate_reading_order(
    block_ids: list[str],
    block_summaries: list[dict[str, Any]],
    page_image_path: str | None = None,
    previous_page_ids: list[str] | None = None,
    model: str | None = None,
    confidence_threshold: float = 0.6,
) -> list[str]:
    """Return block_ids in LLM-validated reading order, or deterministic fallback.

    Args:
        block_ids: Authoritative list of block IDs (only these can appear in output).
        block_summaries: Compact dicts with id, type, bbox, text_preview fields.
        page_image_path: Optional path to a low-res page rendering for multimodal context.
        previous_page_ids: IDs from the end of the previous page (context only).
        model: Ollama model to use; defaults to OLLAMA_MODEL.
        confidence_threshold: Minimum fraction of input IDs that must appear in LLM output.

    Returns:
        Ordered list of block IDs (same set as input).
    """
    if not block_ids:
        return []

    valid_id_set = set(block_ids)
    prev_ids_str = json.dumps((previous_page_ids or [])[-4:])
    blocks_json = json.dumps(
        [_sanitize_summary(s, valid_id_set) for s in block_summaries],
        ensure_ascii=False,
    )

    prompt = _VALIDATION_PROMPT_TEMPLATE.format(
        prev_ids=prev_ids_str,
        blocks_json=blocks_json,
    )

    images: list[str] = []
    if page_image_path:
        encoded = _encode_image(page_image_path)
        if encoded:
            images.append(encoded)

    try:
        raw = _call_ollama(prompt, model or OLLAMA_MODEL, images=images or None)
        candidate_ids = _parse_reading_order(raw, valid_id_set)
        confidence = len(candidate_ids) / len(block_ids) if block_ids else 0.0

        if confidence < confidence_threshold or not _ids_subset_valid(candidate_ids, valid_id_set):
            logger.debug(
                "LLM geometry validator: confidence=%.2f < threshold=%.2f, using deterministic fallback.",
                confidence,
                confidence_threshold,
            )
            return _deterministic_fallback(block_ids, block_summaries)

        # Append any IDs the LLM omitted at the end (preserve completeness).
        missing = [bid for bid in block_ids if bid not in set(candidate_ids)]
        return candidate_ids + missing

    except Exception as exc:
        logger.debug("LLM geometry validation failed (%s), using deterministic fallback.", exc)
        return _deterministic_fallback(block_ids, block_summaries)


def _deterministic_fallback(
    block_ids: list[str],
    block_summaries: list[dict[str, Any]],
) -> list[str]:
    """Preserve the deterministic PageReadingPlan order on LLM failure."""
    return list(block_ids)


def _parse_reading_order(raw: str, valid_ids: set[str]) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting a JSON object from the response.
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return []
        else:
            return []

    order = data.get("reading_order")
    if not isinstance(order, list):
        return []

    seen: set[str] = set()
    result: list[str] = []
    for item in order:
        bid = str(item) if item is not None else ""
        if bid in valid_ids and bid not in seen:
            result.append(bid)
            seen.add(bid)
    return result


def _ids_subset_valid(candidate_ids: list[str], valid_id_set: set[str]) -> bool:
    return all(bid in valid_id_set for bid in candidate_ids)


def _sanitize_summary(summary: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    """Strip any field that is not id/type/bbox/text_preview — prevent data leakage."""
    bid = summary.get("id")
    if bid not in valid_ids:
        return {}
    return {
        "id": bid,
        "type": str(summary.get("type") or ""),
        "bbox": summary.get("bbox"),
        "text_preview": str(summary.get("text_preview") or "")[:80],
    }


def _encode_image(path: str) -> str | None:
    try:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except OSError:
        return None


def _call_ollama(prompt: str, model: str, images: list[str] | None = None) -> str:
    payload_data: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": OLLAMA_OPTIONS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if images:
        payload_data["images"] = images

    payload = json.dumps(payload_data).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read())
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str(data.get("response", ""))
