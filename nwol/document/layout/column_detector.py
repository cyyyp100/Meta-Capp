from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Protocol

from document.models import BoundingBox, RawBlock


class HasBBox(Protocol):
    bbox: BoundingBox | None
    page: int | None


@dataclass(slots=True)
class ColumnLayout:
    layout_type: str
    columns: list[list[HasBBox]] = field(default_factory=list)
    full_width_blocks: list[HasBBox] = field(default_factory=list)
    page_width: float = 0.0


def detect_columns(
    blocks: list[HasBBox],
    page_width: float | None = None,
    min_per_column: int = 2,
) -> ColumnLayout:
    valid = [block for block in blocks if block.bbox is not None]
    if not valid:
        return ColumnLayout("single_column", [blocks[:]], [], page_width or 0.0)

    width = page_width or max((block.bbox.x1 for block in valid if block.bbox), default=0.0)
    if width <= 0:
        return ColumnLayout("single_column", [blocks[:]], [], 0.0)

    full_width = [block for block in valid if _is_full_width(block, width)]
    full_width_ids = {id(block) for block in full_width}
    candidates = [block for block in valid if id(block) not in full_width_ids]
    if len(candidates) < min_per_column * 2:
        return ColumnLayout("single_column", [valid], full_width, width)

    centers = sorted(block.bbox.center_x for block in candidates if block.bbox)
    if len(centers) < min_per_column * 2:
        return ColumnLayout("single_column", [valid], full_width, width)

    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    max_gap, split_index = max(gaps, default=(0.0, 0))
    center_span = max(centers) - min(centers)
    central_gap = centers[split_index] < width * 0.48 and centers[split_index + 1] > width * 0.52
    # Figure labels, narrow formulas, and full-width captions can stretch the
    # observed center span just enough to hide an otherwise clear gutter. Keep
    # the threshold conservative, but leave a little room for those outliers.
    enough_gap = max_gap >= max(width * 0.10, center_span * 0.22)
    if not central_gap and not enough_gap:
        robust = _detect_columns_from_body_lines(candidates, full_width, width, min_per_column)
        if robust is not None:
            return robust
        return ColumnLayout("single_column", [valid], full_width, width)

    split = (centers[split_index] + centers[split_index + 1]) / 2.0
    left = [block for block in candidates if block.bbox and block.bbox.center_x <= split]
    right = [block for block in candidates if block.bbox and block.bbox.center_x > split]
    if len(left) < min_per_column or len(right) < min_per_column:
        robust = _detect_columns_from_body_lines(candidates, full_width, width, min_per_column)
        if robust is not None:
            return robust
        return ColumnLayout("single_column", [valid], full_width, width)
    if min(len(left), len(right)) / max(len(left), len(right)) < 0.35:
        robust = _detect_columns_from_body_lines(candidates, full_width, width, min_per_column)
        if robust is not None:
            return robust
        return ColumnLayout("single_column", [valid], full_width, width)
    if not _columns_overlap_vertically(left, right):
        robust = _detect_columns_from_body_lines(candidates, full_width, width, min_per_column)
        if robust is not None:
            return robust
        return ColumnLayout("single_column", [valid], full_width, width)
    if not (_column_has_textual_content(left) and _column_has_textual_content(right)):
        robust = _detect_columns_from_body_lines(candidates, full_width, width, min_per_column)
        if robust is not None:
            return robust
        return ColumnLayout("single_column", [valid], full_width, width)

    left_width = median([block.bbox.width for block in left if block.bbox] or [0.0])
    right_width = median([block.bbox.width for block in right if block.bbox] or [0.0])
    if left_width <= 0 or right_width <= 0:
        return ColumnLayout("single_column", [valid], full_width, width)

    return ColumnLayout("two_columns", [left, right], full_width, width)


def _columns_overlap_vertically(left: list[HasBBox], right: list[HasBBox]) -> bool:
    left_y0 = min(block.bbox.y0 for block in left if block.bbox)
    left_y1 = max(block.bbox.y1 for block in left if block.bbox)
    right_y0 = min(block.bbox.y0 for block in right if block.bbox)
    right_y1 = max(block.bbox.y1 for block in right if block.bbox)
    return min(left_y1, right_y1) - max(left_y0, right_y0) > 0


def _column_has_textual_content(blocks: list[HasBBox]) -> bool:
    for block in blocks:
        block_type = str(getattr(block, "type", "") or getattr(block, "block_type", "") or "").casefold()
        if block_type in {"figure", "table", "formula"}:
            continue
        metadata = getattr(block, "metadata", None) or {}
        if isinstance(metadata, dict) and metadata.get("is_caption"):
            continue
        text = str(getattr(block, "text", "") or getattr(block, "caption", "") or "").strip()
        if len(_WORD_RE.findall(text)) >= 1:
            return True
    return False


def _detect_columns_from_body_lines(
    candidates: list[HasBBox],
    full_width: list[HasBBox],
    page_width: float,
    min_per_column: int,
) -> ColumnLayout | None:
    """Detect mixed scientific pages where diagrams/formulas fill the center gap."""
    body = [block for block in candidates if _looks_like_body_column_candidate(block, page_width)]
    if len(body) < min_per_column * 2:
        return None

    centers = sorted(block.bbox.center_x for block in body if block.bbox)
    if len(centers) < min_per_column * 2:
        return None

    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    max_gap, split_index = max(gaps, default=(0.0, 0))
    center_span = max(centers) - min(centers)
    if max_gap < max(page_width * 0.11, center_span * 0.22):
        return None

    split = (centers[split_index] + centers[split_index + 1]) / 2.0
    if not (page_width * 0.34 <= split <= page_width * 0.66):
        return None

    body_left = [block for block in body if block.bbox and block.bbox.center_x <= split]
    body_right = [block for block in body if block.bbox and block.bbox.center_x > split]
    if len(body_left) < min_per_column or len(body_right) < min_per_column:
        return None
    if min(len(body_left), len(body_right)) / max(len(body_left), len(body_right)) < 0.25:
        return None
    if not _columns_overlap_vertically(body_left, body_right):
        return None

    left_width = median([block.bbox.width for block in body_left if block.bbox] or [0.0])
    right_width = median([block.bbox.width for block in body_right if block.bbox] or [0.0])
    if left_width < page_width * 0.22 or right_width < page_width * 0.22:
        return None

    left: list[HasBBox] = []
    right: list[HasBBox] = []
    extra_full_width = list(full_width)
    for block in candidates:
        if block.bbox is None:
            continue
        if _crosses_detected_gutter(block, split, page_width, left_width, right_width):
            extra_full_width.append(block)
        elif block.bbox.center_x <= split:
            left.append(block)
        else:
            right.append(block)

    if len(left) < min_per_column or len(right) < min_per_column:
        return None
    return ColumnLayout("two_columns", [left, right], extra_full_width, page_width)


_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{2,}\b")
_MATHISH_RE = re.compile(r"\\[A-Za-z]+|[_^{}=<>]|[∑∫√∞≈≠≤≥→←↔∈∉∀∃αβγδλμσφψω]")


def _is_full_width(block: HasBBox, page_width: float) -> bool:
    if not block.bbox:
        return False
    # Single-column scientific pages often use wide text blocks with short
    # section headings and final line fragments. Treat those wide body blocks
    # as full-width so they do not get split into a fake right column.
    if block.bbox.width >= page_width * 0.60:
        return True
    text = str(getattr(block, "text", "") or getattr(block, "caption", "") or "").strip()
    centered = abs(block.bbox.center_x - page_width / 2.0) <= page_width * 0.08
    return bool(
        centered
        and block.bbox.y0 <= 220.0
        and page_width * 0.16 <= block.bbox.width <= page_width * 0.60
        and len(_WORD_RE.findall(text)) >= 2
    )


def _looks_like_body_column_candidate(block: HasBBox, page_width: float) -> bool:
    bbox = block.bbox
    if bbox is None:
        return False
    if bbox.width < page_width * 0.18 or bbox.width > page_width * 0.54:
        return False
    if bbox.height > max(42.0, bbox.width * 0.45):
        return False

    block_type = str(getattr(block, "type", "") or getattr(block, "block_type", "") or "").casefold()
    if block_type in {"figure", "table", "formula"}:
        return False

    text = str(getattr(block, "text", "") or getattr(block, "caption", "") or "").strip()
    words = _WORD_RE.findall(text)
    if len(words) < 3:
        return False
    math_signals = len(_MATHISH_RE.findall(text))
    return math_signals <= max(3, len(words))


def _crosses_detected_gutter(
    block: HasBBox,
    split: float,
    page_width: float,
    left_width: float,
    right_width: float,
) -> bool:
    bbox = block.bbox
    if bbox is None:
        return False
    gutter_pad = max(8.0, page_width * 0.018)
    if not (bbox.x0 < split - gutter_pad and bbox.x1 > split + gutter_pad):
        return False
    typical_column_width = min(left_width, right_width)
    return bbox.width >= max(page_width * 0.34, typical_column_width * 0.78)


def raw_lines_to_blocks(lines: list, block_type: str = "line") -> list[RawBlock]:
    return [
        RawBlock(text=line.text, bbox=line.bbox, page=line.page, block_type=block_type, lines=[line])
        for line in lines
        if line.text.strip()
    ]
