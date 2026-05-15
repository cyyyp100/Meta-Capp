from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from document.layout.column_detector import ColumnLayout, detect_columns
from document.layout.reading_order import order_page_blocks, _stamp_reading_order
from document.models import BoundingBox


class HasBBoxAndId(Protocol):
    bbox: BoundingBox | None
    page: int | None
    id: str | None
    type: str


@dataclass(slots=True)
class PageReadingPlan:
    """Source of truth for reading order, section boundaries, and figure placement on one page."""

    page_number: int
    # block IDs per detected column (empty list for single-column pages)
    columns: list[list[str]] = field(default_factory=list)
    # block IDs of full-width blocks in document order
    full_width_blocks: list[str] = field(default_factory=list)
    # ordered sequence of all displayable block IDs (the canonical reading order)
    reading_order_ids: list[str] = field(default_factory=list)
    # block IDs of figures, tables, formulas (anchors that interrupt text flow)
    visual_anchors: list[str] = field(default_factory=list)
    # block IDs that were dropped (headers/footers, metadata, empty)
    ignored_blocks: list[str] = field(default_factory=list)
    # block IDs that continue content from the previous page (detected by missing leading capital)
    continuations: list[str] = field(default_factory=list)


_VISUAL_TYPES = {"figure", "table", "formula"}
_IGNORED_METADATA_KEYS = ("is_header_footer", "is_metadata", "is_reference")


def build_page_reading_plan(
    page_number: int,
    blocks: list[HasBBoxAndId],
    page_width: float | None = None,
) -> PageReadingPlan:
    """Build a PageReadingPlan from the blocks belonging to a single page."""
    if not blocks:
        return PageReadingPlan(page_number=page_number)

    ignored: list[str] = []
    active: list[HasBBoxAndId] = []
    for block in blocks:
        if _should_ignore(block):
            if block.id:
                ignored.append(block.id)
        else:
            active.append(block)

    layout: ColumnLayout = detect_columns(active, page_width=page_width)  # type: ignore[arg-type]
    ordered = order_page_blocks(active, layout)  # type: ignore[arg-type]
    _stamp_reading_order(ordered)  # type: ignore[arg-type]

    reading_order_ids = [b.id for b in ordered if b.id]
    visual_anchors = [b.id for b in ordered if b.id and b.type in _VISUAL_TYPES]

    columns: list[list[str]] = []
    if layout.layout_type == "two_columns":
        for col in layout.columns:
            columns.append([b.id for b in col if b.id])

    full_width_ids = [b.id for b in layout.full_width_blocks if b.id]
    continuations = _detect_continuations(active)

    return PageReadingPlan(
        page_number=page_number,
        columns=columns,
        full_width_blocks=full_width_ids,
        reading_order_ids=reading_order_ids,
        visual_anchors=visual_anchors,
        ignored_blocks=ignored,
        continuations=continuations,
    )


def build_document_reading_plans(
    blocks: list[HasBBoxAndId],
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> dict[int, PageReadingPlan]:
    """Build one PageReadingPlan per page from a flat block list."""
    page_sizes = page_sizes or {}
    by_page: dict[int, list[HasBBoxAndId]] = {}
    for block in blocks:
        p = int(block.page or 0)
        by_page.setdefault(p, []).append(block)

    return {
        page: build_page_reading_plan(
            page,
            page_blocks,
            page_width=page_sizes.get(page, (0.0, 0.0))[0] or None,
        )
        for page, page_blocks in sorted(by_page.items())
    }


def reorder_blocks_by_ids(blocks: list[HasBBoxAndId], reading_order_ids: list[str]) -> list[HasBBoxAndId]:
    """Return blocks ordered by a validated ID permutation, preserving missing IDs at the end."""
    if not blocks or not reading_order_ids:
        return list(blocks)

    by_id = {block.id: block for block in blocks if block.id}
    seen: set[str] = set()
    ordered: list[HasBBoxAndId] = []
    for block_id in reading_order_ids:
        if block_id in seen:
            continue
        block = by_id.get(block_id)
        if block is None:
            continue
        ordered.append(block)
        seen.add(block_id)

    ordered_ids = {id(block) for block in ordered}
    ordered.extend(block for block in blocks if id(block) not in ordered_ids)
    return ordered


def page_reading_plan_to_dict(plan: PageReadingPlan) -> dict:
    return {
        "page_number": plan.page_number,
        "columns": [list(column) for column in plan.columns],
        "full_width_blocks": list(plan.full_width_blocks),
        "reading_order_ids": list(plan.reading_order_ids),
        "visual_anchors": list(plan.visual_anchors),
        "ignored_blocks": list(plan.ignored_blocks),
        "continuations": list(plan.continuations),
    }


def _should_ignore(block: HasBBoxAndId) -> bool:
    metadata = getattr(block, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return False
    return any(metadata.get(key) for key in _IGNORED_METADATA_KEYS)


def _detect_continuations(blocks: list[HasBBoxAndId]) -> list[str]:
    """Blocks whose text starts mid-sentence (no leading capital/number), likely page-break continuations."""
    result: list[str] = []
    for block in blocks:
        if not block.id:
            continue
        text = str(getattr(block, "text", "") or "").lstrip()
        if text and text[0].islower():
            result.append(block.id)
    return result
