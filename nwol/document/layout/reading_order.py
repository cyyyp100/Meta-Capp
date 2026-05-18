from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from document.layout.column_detector import ColumnLayout, detect_columns
from document.models import BoundingBox


class HasBBox(Protocol):
    bbox: BoundingBox | None
    page: int | None


def order_blocks_for_reading(
    blocks: list[HasBBox],
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> list[HasBBox]:
    page_sizes = page_sizes or {}
    by_page: dict[int, list[HasBBox]] = defaultdict(list)
    without_page: list[HasBBox] = []
    for block in blocks:
        if block.page is None:
            without_page.append(block)
        else:
            by_page[int(block.page)].append(block)

    ordered: list[HasBBox] = []
    for page in sorted(by_page):
        page_blocks = by_page[page]
        page_width = page_sizes.get(page, (0.0, 0.0))[0] or None
        layout = detect_columns(page_blocks, page_width=page_width)
        ordered.extend(order_page_blocks(page_blocks, layout))
    ordered.extend(sorted(without_page, key=_position_key))
    _stamp_reading_order(ordered)
    return ordered


def order_page_blocks(blocks: list[HasBBox], layout: ColumnLayout | None = None) -> list[HasBBox]:
    if not blocks:
        return []
    layout = layout or detect_columns(blocks)
    if layout.layout_type != "two_columns":
        return _sort_blocks_top_left(blocks)

    # Use column membership from detect_columns, not a page_width/2 re-split.
    left_ids = {id(b) for b in layout.columns[0]} if len(layout.columns) > 0 else set()
    right_ids = {id(b) for b in layout.columns[1]} if len(layout.columns) > 1 else set()
    narrow = [b for b in blocks if id(b) in left_ids or id(b) in right_ids]
    full_width = _sort_blocks_top_left(layout.full_width_blocks)
    result: list[HasBBox] = []
    remaining = {id(b) for b in narrow}

    for wide in full_width:
        if wide.bbox is None:
            continue
        before_left = [
            b for b in narrow
            if id(b) in remaining and id(b) in left_ids
            and b.bbox is not None and b.bbox.y0 < wide.bbox.y0
        ]
        before_right = [
            b for b in narrow
            if id(b) in remaining and id(b) in right_ids
            and b.bbox is not None and b.bbox.y0 < wide.bbox.y0
        ]
        result.extend(_sort_blocks_top_left(before_left))
        result.extend(_sort_blocks_top_left(before_right))
        for b in before_left + before_right:
            remaining.discard(id(b))
        result.append(wide)

    rest_left = [b for b in narrow if id(b) in remaining and id(b) in left_ids]
    rest_right = [b for b in narrow if id(b) in remaining and id(b) in right_ids]
    result.extend(_sort_blocks_top_left(rest_left))
    result.extend(_sort_blocks_top_left(rest_right))

    placed = {id(b) for b in result}
    for b in _sort_blocks_top_left(blocks):
        if id(b) not in placed:
            result.append(b)
    return result


def _order_two_columns(blocks: list[HasBBox], page_width: float) -> list[HasBBox]:
    """Kept for external callers; order_page_blocks uses column membership directly."""
    if not blocks:
        return []
    if page_width <= 0:
        page_width = max((block.bbox.x1 for block in blocks if block.bbox), default=0.0)
    left = [block for block in blocks if block.bbox and block.bbox.center_x < page_width / 2.0]
    right = [block for block in blocks if block.bbox and block.bbox.center_x >= page_width / 2.0]
    return sorted(left, key=_position_key) + sorted(right, key=_position_key)


def _stamp_reading_order(blocks: list[HasBBox]) -> None:
    for index, block in enumerate(blocks):
        metadata = getattr(block, "metadata", None)
        if isinstance(metadata, dict):
            metadata["reading_order_index"] = index


def _sort_blocks_top_left(blocks: list[HasBBox]) -> list[HasBBox]:
    """Sort visual rows top-to-bottom, then left-to-right inside a row."""
    if len(blocks) <= 1:
        return list(blocks)

    with_bbox = [block for block in blocks if block.bbox is not None]
    without_bbox = [block for block in blocks if block.bbox is None]
    rows: list[list[HasBBox]] = []
    for block in sorted(with_bbox, key=_position_key):
        for row in rows[-6:]:
            if _block_fits_visual_row(row, block):
                row.append(block)
                break
        else:
            rows.append([block])

    ordered: list[HasBBox] = []
    for row in sorted(rows, key=lambda item: _position_key(item[0])):
        ordered.extend(sorted(row, key=lambda item: (int(item.page or 0), item.bbox.x0 if item.bbox else float("inf"))))
    ordered.extend(sorted(without_bbox, key=_position_key))
    return ordered


def _block_fits_visual_row(row: list[HasBBox], block: HasBBox) -> bool:
    if not row or block.bbox is None:
        return False
    page = int(block.page or 0)
    if any(int(item.page or 0) != page for item in row):
        return False
    bbox = _row_bbox(row)
    if bbox is None:
        return False
    overlap = min(bbox.y1, block.bbox.y1) - max(bbox.y0, block.bbox.y0)
    min_height = max(1.0, min(bbox.height, block.bbox.height))
    if overlap >= min_height * 0.28:
        return True
    center_delta = abs(bbox.center_y - block.bbox.center_y)
    return center_delta <= max(5.5, min_height * 0.65)


def _row_bbox(row: list[HasBBox]) -> BoundingBox | None:
    bbox = row[0].bbox if row else None
    for block in row[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    return bbox


def _position_key(block: HasBBox) -> tuple[int, float, float]:
    page = int(block.page or 0)
    if block.bbox is None:
        return (page, float("inf"), float("inf"))
    return (page, block.bbox.y0, block.bbox.x0)
