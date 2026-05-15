from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from document.layout.page_reading_plan import PageReadingPlan
from document.models import BoundingBox
from document.postprocess.latex_quality import latex_looks_corrupt


class HasLayoutFields(Protocol):
    bbox: BoundingBox | None
    page: int | None
    id: str | None
    type: str
    text: str
    metadata: dict


@dataclass(slots=True)
class LayoutRisk:
    score: float
    reasons: list[str] = field(default_factory=list)
    needs_llm_order: bool = False
    needs_llm_crop: bool = False
    needs_llm_latex: bool = False

    def to_dict(self) -> dict:
        return {
            "score": round(float(self.score), 4),
            "reasons": list(self.reasons),
            "needs_llm_order": bool(self.needs_llm_order),
            "needs_llm_crop": bool(self.needs_llm_crop),
            "needs_llm_latex": bool(self.needs_llm_latex),
        }


@dataclass(slots=True)
class CropRisk:
    score: float
    reasons: list[str] = field(default_factory=list)
    needs_llm: bool = False

    def to_dict(self) -> dict:
        return {
            "score": round(float(self.score), 4),
            "reasons": list(self.reasons),
            "needs_llm": bool(self.needs_llm),
        }


def compute_layout_risk(
    blocks: list[HasLayoutFields],
    plan: PageReadingPlan,
    quality_warnings: list[str] | None = None,
    document_type: str | None = None,
    prev_page_tail: list[str] | None = None,
) -> LayoutRisk:
    score = 0.0
    reasons: list[str] = []

    if plan.columns:
        score += 0.25
        reasons.append("two_columns")

    if _has_full_width_block_between_columns(blocks, plan):
        score += 0.18
        reasons.append("full_width_between_columns")

    if len(plan.visual_anchors) >= 2:
        score += 0.15
        reasons.append("many_visual_anchors")
    elif len(plan.visual_anchors) == 1 and _visual_anchor_between_text(blocks, plan):
        score += 0.10
        reasons.append("visual_anchor_between_text")

    warnings_text = " ".join(quality_warnings or []).casefold()
    if "colonne" in warnings_text or "column" in warnings_text:
        score += 0.25
        reasons.append("quality_mixed_columns")
    if "faible" in warnings_text or "low" in warnings_text:
        score += 0.10
        reasons.append("quality_low_score")

    if has_vertical_regression(blocks):
        score += 0.20
        reasons.append("vertical_regression")

    if _has_parallel_bands(blocks):
        score += 0.12
        reasons.append("near_vertical_ties")

    if any(block.bbox is None or block.page is None for block in blocks):
        score += 0.12
        reasons.append("missing_geometry")

    if prev_page_tail and plan.continuations:
        score += 0.10
        reasons.append("cross_page_continuation")

    needs_latex = any(needs_latex_llm(block) for block in blocks)
    if needs_latex:
        score += 0.08
        reasons.append("latex_ambiguous")

    is_scientific = _is_scientific(document_type)
    needs_order = bool(plan.columns) or score > 0.45 or (score >= 0.35 and is_scientific)

    return LayoutRisk(
        score=min(1.0, score),
        reasons=_unique(reasons),
        needs_llm_order=needs_order,
        needs_llm_crop=any(compute_crop_risk(block, blocks).needs_llm for block in blocks if block.type in _VISUAL_TYPES),
        needs_llm_latex=needs_latex,
    )


def compute_crop_risk(block: HasLayoutFields, page_blocks: list[HasLayoutFields]) -> CropRisk:
    score = 0.0
    reasons: list[str] = []
    bbox = block.bbox
    metadata = block.metadata or {}

    if block.type not in _VISUAL_TYPES:
        return CropRisk(0.0, [], False)

    if bbox is None:
        score += 0.55
        reasons.append("missing_bbox")
    else:
        page_width = _metadata_float(metadata, "page_width")
        page_height = _metadata_float(metadata, "page_height")
        if bbox.width < 24.0 or bbox.height < 16.0:
            score += 0.25
            reasons.append("bbox_too_tight")
        if page_width and bbox.width > page_width * 0.92:
            score += 0.22
            reasons.append("bbox_too_wide")
        if page_height and bbox.height > page_height * 0.70:
            score += 0.22
            reasons.append("bbox_too_tall")

    if block.type == "figure" and not str(getattr(block, "caption", "") or block.text or "").strip():
        if _nearby_caption_exists(block, page_blocks):
            score += 0.22
            reasons.append("near_caption_unattached")

    if _nearby_visual_exists(block, page_blocks):
        score += 0.12
        reasons.append("nearby_visual")

    source = str(metadata.get("source") or metadata.get("geometry_source") or "")
    if "vector" in source or "drawing" in source:
        score += 0.25
        reasons.append("vector_reconstructed")

    if block.type == "formula" and not (
        metadata.get("formula_image_path") or getattr(block, "image_path", None) or getattr(block, "latex", None)
    ):
        score += 0.20
        reasons.append("formula_without_render")

    if metadata.get("context_asset_reason") in {"inline_math", "display_math", "math_dense_text", "fragmented_math_text"}:
        score += 0.12
        reasons.append("math_context_asset")

    score = min(1.0, score)
    return CropRisk(score=score, reasons=_unique(reasons), needs_llm=score >= 0.45)


def needs_latex_llm(block: HasLayoutFields) -> bool:
    if block.type not in {"paragraph", "formula", "definition", "theorem", "example", "remark"}:
        return False
    metadata = block.metadata or {}
    text = str(getattr(block, "latex", None) or block.text or "")
    if not text.strip():
        return False
    if metadata.get("context_asset_path") and _looks_math_dense(text):
        return True
    if metadata.get("formula_mode") == "ambiguous":
        return True
    if text.count("$") % 2 == 1:
        return True
    if latex_looks_corrupt(text):
        return True
    if re.search(r"\\[A-Za-z]*$|[_^]\s*$|[({\[]\s*$", text):
        return True
    return _looks_math_dense(text) and (metadata.get("contains_inline_math") or len(text) <= 240)


def has_vertical_regression(blocks: list[HasLayoutFields]) -> bool:
    previous_y = None
    previous_x = None
    for block in blocks:
        bbox = block.bbox
        if bbox is None:
            continue
        if previous_y is not None and bbox.y0 + 6.0 < previous_y:
            if previous_x is None or abs(bbox.x0 - previous_x) > 80.0:
                return True
        previous_y = bbox.y0
        previous_x = bbox.x0
    return False


_VISUAL_TYPES = {"figure", "table", "formula"}
_CAPTION_RE = re.compile(r"^(?:figure|fig\.?|schema|schéma|graphique|diagramme|tableau|table)\b", re.I)
_MATH_SIGNAL_RE = re.compile(r"\\[A-Za-z]+|[_^{}=<>]|[∑∫√∞≈≠≤≥→←↔∈∉∀∃αβγδλμσφψω]")
_PROSE_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{3,}\b")


def _has_full_width_block_between_columns(blocks: list[HasLayoutFields], plan: PageReadingPlan) -> bool:
    if not plan.columns or not plan.full_width_blocks:
        return False
    y_values = [block.bbox.y0 for block in blocks if block.bbox and block.id in set(sum(plan.columns, []))]
    if not y_values:
        return False
    top, bottom = min(y_values), max(y_values)
    by_id = {block.id: block for block in blocks if block.id}
    return any(
        (by_id.get(block_id) is not None and by_id[block_id].bbox is not None and top < by_id[block_id].bbox.y0 < bottom)
        for block_id in plan.full_width_blocks
    )


def _visual_anchor_between_text(blocks: list[HasLayoutFields], plan: PageReadingPlan) -> bool:
    order = plan.reading_order_ids
    by_id = {block.id: block for block in blocks if block.id}
    for index, block_id in enumerate(order):
        block = by_id.get(block_id)
        if block is None or block.type not in _VISUAL_TYPES:
            continue
        before = any((by_id.get(bid) is not None and by_id[bid].type == "paragraph") for bid in order[:index])
        after = any((by_id.get(bid) is not None and by_id[bid].type == "paragraph") for bid in order[index + 1 :])
        if before and after:
            return True
    return False


def _has_parallel_bands(blocks: list[HasLayoutFields]) -> bool:
    valid = [block for block in blocks if block.bbox is not None]
    for left_index, left in enumerate(valid):
        assert left.bbox is not None
        for right in valid[left_index + 1 :]:
            assert right.bbox is not None
            if abs(left.bbox.y0 - right.bbox.y0) <= 4.0 and abs(left.bbox.x0 - right.bbox.x0) > 90.0:
                return True
    return False


def _nearby_caption_exists(block: HasLayoutFields, page_blocks: list[HasLayoutFields]) -> bool:
    if block.bbox is None:
        return False
    for other in page_blocks:
        if other is block or other.page != block.page or other.bbox is None:
            continue
        if not _CAPTION_RE.match(str(other.text or "").strip()):
            continue
        if abs(other.bbox.y0 - block.bbox.y1) <= 180.0 or abs(block.bbox.y0 - other.bbox.y1) <= 180.0:
            return True
    return False


def _nearby_visual_exists(block: HasLayoutFields, page_blocks: list[HasLayoutFields]) -> bool:
    if block.bbox is None:
        return False
    for other in page_blocks:
        if other is block or other.type not in _VISUAL_TYPES or other.page != block.page or other.bbox is None:
            continue
        vertical_gap = max(other.bbox.y0 - block.bbox.y1, block.bbox.y0 - other.bbox.y1, 0.0)
        horizontal_overlap = min(block.bbox.x1, other.bbox.x1) - max(block.bbox.x0, other.bbox.x0)
        if vertical_gap <= 48.0 and horizontal_overlap > 0:
            return True
    return False


def _looks_math_dense(text: str) -> bool:
    signals = len(_MATH_SIGNAL_RE.findall(text or ""))
    prose = len(_PROSE_RE.findall(text or ""))
    return signals >= 4 or (signals >= 2 and prose <= 4)


def _metadata_float(metadata: dict, key: str) -> float | None:
    try:
        value = float(metadata.get(key) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _is_scientific(document_type: str | None) -> bool:
    return "scientific" in str(document_type or "").casefold() or "article" in str(document_type or "").casefold()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
