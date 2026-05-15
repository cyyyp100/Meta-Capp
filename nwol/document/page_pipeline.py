from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from document.extractors.pymupdf_extractor import PyMuPDFExtractor
from document.layout.llm_order import llm_order_with_cache
from document.layout.page_reading_plan import (
    PageReadingPlan,
    build_page_reading_plan,
    page_reading_plan_to_dict,
    reorder_blocks_by_ids,
)
from document.layout.risk import LayoutRisk, compute_crop_risk, compute_layout_risk, needs_latex_llm
from document.models import BoundingBox, DocumentBlock
from document.postprocess.latex_quality import latex_looks_corrupt
from llm.pdf_assistant_queue import PDF_LLM_PRIORITIES, get_pdf_llm_queue
from config.settings import LLM_CROP_TIMEOUT

logger = logging.getLogger("Document.page_pipeline")


@dataclass(slots=True)
class PageExtractionResult:
    pdf_path: str
    page_number: int
    blocks: list[DocumentBlock]
    page_plan: PageReadingPlan
    layout_risk: LayoutRisk
    score: float
    warnings: list[str]
    engine_name: str = PyMuPDFExtractor.engine_name
    page_size: tuple[float, float] | None = None
    enrich_assets: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_reader_blocks(self) -> list[dict[str, Any]]:
        return [block.to_reader_dict() for block in self.blocks]

    def page_plan_dict(self) -> dict:
        return page_reading_plan_to_dict(self.page_plan)

    def layout_risk_dict(self) -> dict:
        return self.layout_risk.to_dict()


def extract_page_lazy(
    pdf_path: str,
    page_number: int,
    prev_page_tail: list[str] | None = None,
    enrich_assets: bool = True,
    *,
    document_type: str | None = None,
    validate_with_llm: bool = True,
    llm_generation: int | None = None,
) -> PageExtractionResult:
    """Single entry point for reader-facing page extraction."""
    logger.info("[PAGE_EXTRACT] start path=%s page=%s enrich_assets=%s", pdf_path, page_number, enrich_assets)
    extraction = PyMuPDFExtractor().extract_page(
        pdf_path,
        page_number,
        enrich_assets=enrich_assets,
    )
    blocks = [block for block in extraction.blocks if int(block.page or page_number) == page_number]
    _ensure_block_ids(blocks, page_number)

    page_sizes = extraction.metadata.get("page_sizes") if isinstance(extraction.metadata, dict) else {}
    page_size = page_sizes.get(page_number) if isinstance(page_sizes, dict) else None
    page_width = (page_size or (0.0, 0.0))[0] or _infer_page_width(blocks)

    plan = build_page_reading_plan(page_number=page_number, blocks=blocks, page_width=page_width)
    blocks = reorder_blocks_by_ids(blocks, plan.reading_order_ids)
    _stamp_plan_metadata(blocks, plan)
    logger.info("[PAGE_PLAN] page=%s blocs=%s anchors=%s", page_number, len(blocks), len(plan.visual_anchors))

    risk = compute_layout_risk(
        blocks=blocks,
        plan=plan,
        quality_warnings=extraction.warnings,
        document_type=document_type,
        prev_page_tail=prev_page_tail,
    )
    logger.info("[LAYOUT_RISK] page=%s score=%.2f reasons=%s", page_number, risk.score, ",".join(risk.reasons))

    if validate_with_llm and risk.needs_llm_order:
        order_ids = llm_order_with_cache(
            pdf_path=pdf_path,
            page_number=page_number,
            blocks=blocks,
            plan=plan,
            prev_page_tail=prev_page_tail,
            generation=llm_generation,
        )
        blocks = reorder_blocks_by_ids(blocks, order_ids)
        plan.reading_order_ids = [block.id for block in blocks if block.id]
        _stamp_plan_metadata(blocks, plan)
        _stamp_llm_order_status(blocks, "attempted")
    elif risk.needs_llm_order:
        _stamp_llm_order_status(blocks, "pending")

    if validate_with_llm and enrich_assets and risk.needs_llm_crop:
        refined = _try_refine_risky_crops(
            pdf_path,
            page_number,
            blocks,
            generation=llm_generation,
        )
        if refined is not None:
            blocks = refined
            _ensure_block_ids(blocks, page_number)
            plan = build_page_reading_plan(page_number=page_number, blocks=blocks, page_width=page_width)
            blocks = reorder_blocks_by_ids(blocks, plan.reading_order_ids)
            _stamp_plan_metadata(blocks, plan)
            if risk.needs_llm_order:
                _stamp_llm_order_status(blocks, "attempted")
        _stamp_llm_crop_status(blocks, "attempted")
    elif enrich_assets and risk.needs_llm_crop:
        _stamp_llm_crop_status(blocks, "pending")

    _annotate_crop_and_latex_risk(blocks)
    reader_blocks = _normalize_reader_types(blocks)

    return PageExtractionResult(
        pdf_path=pdf_path,
        page_number=page_number,
        blocks=reader_blocks,
        page_plan=plan,
        layout_risk=risk,
        score=extraction.score,
        warnings=extraction.warnings,
        engine_name=extraction.engine_name,
        page_size=page_size,
        enrich_assets=enrich_assets,
        metadata={
            "page_plan": page_reading_plan_to_dict(plan),
            "layout_risk": risk.to_dict(),
        },
    )


_READER_TYPES = {
    "heading",
    "subheading",
    "subsubheading",
    "paragraph",
    "formula",
    "table",
    "figure",
    "bullet_list",
    "definition",
    "theorem",
    "example",
    "remark",
    "warning",
    "abstract",
}


def _ensure_block_ids(blocks: list[DocumentBlock], page_number: int) -> None:
    used: set[str] = set()
    for index, block in enumerate(blocks, start=1):
        candidate = block.id or f"p{page_number}_b{index}"
        if candidate in used:
            candidate = f"p{page_number}_b{index}"
        block.id = candidate
        used.add(candidate)


def _infer_page_width(blocks: list[DocumentBlock]) -> float | None:
    widths = []
    for block in blocks:
        try:
            width = float((block.metadata or {}).get("page_width") or 0.0)
        except (TypeError, ValueError):
            width = 0.0
        if width > 0:
            widths.append(width)
        elif block.bbox is not None:
            widths.append(block.bbox.x1)
    return max(widths) if widths else None


def _stamp_plan_metadata(blocks: list[DocumentBlock], plan: PageReadingPlan) -> None:
    order_index = {block_id: index for index, block_id in enumerate(plan.reading_order_ids)}
    for block in blocks:
        if block.id in order_index:
            block.metadata["reading_order_index"] = order_index[block.id]
        block.metadata["page_plan_id"] = f"p{plan.page_number}"
        if block.id in plan.visual_anchors:
            block.metadata["visual_anchor"] = True
        if block.id in plan.continuations:
            block.metadata["page_continuation"] = True


def _stamp_llm_order_status(blocks: list[DocumentBlock], status: str) -> None:
    for block in blocks:
        block.metadata["llm_order_status"] = status


def _stamp_llm_crop_status(blocks: list[DocumentBlock], status: str) -> None:
    for block in blocks:
        if block.type in {"figure", "table", "formula"} or block.image_path:
            block.metadata["llm_crop_status"] = status


def _annotate_crop_and_latex_risk(blocks: list[DocumentBlock]) -> None:
    for block in blocks:
        if block.type in {"figure", "table", "formula"}:
            crop_risk = compute_crop_risk(block, blocks)
            if crop_risk.score > 0:
                block.metadata["crop_risk"] = crop_risk.to_dict()
        if block.type == "formula" and latex_looks_corrupt(block.latex or block.text):
            block.metadata["latex_corrupt"] = True
            block.metadata["needs_latex_llm"] = True
            block.confidence = min(float(block.confidence or 1.0), 0.55)
        if needs_latex_llm(block):
            block.metadata["needs_latex_llm"] = True


def _normalize_reader_types(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for block in blocks:
        if block.type not in _READER_TYPES:
            block.type = "paragraph" if block.text else "figure" if block.image_path else "paragraph"
        result.append(block)
    return result


def _try_refine_risky_crops(
    pdf_path: str,
    page_number: int,
    blocks: list[DocumentBlock],
    *,
    generation: int | None,
) -> list[DocumentBlock] | None:
    try:
        from document.postprocess.llm_page_cropper import llm_crop_page_formulas
    except Exception as exc:
        logger.debug("[LLM_CROP] module indisponible: %s", exc)
        return None

    work_blocks = [_copy_block(block) for block in blocks]

    def _call() -> list[DocumentBlock]:
        return llm_crop_page_formulas(
            pdf_path,
            work_blocks,
            max_pages=1,
            pages={page_number},
        )

    result = get_pdf_llm_queue().run_sync(
        "crop_visible",
        _call,
        priority=PDF_LLM_PRIORITIES["crop_visible"],
        timeout=LLM_CROP_TIMEOUT,
        generation=generation,
    )
    if result is None:
        logger.info("[LLM_CROP] fallback PyMuPDF page=%s", page_number)
        return None
    logger.info("[LLM_CROP] page=%s terminé", page_number)
    return result


def _copy_block(block: DocumentBlock) -> DocumentBlock:
    return DocumentBlock(
        type=block.type,
        text=block.text,
        page=block.page,
        bbox=BoundingBox.from_seq(block.bbox.to_list()) if block.bbox else None,
        level=block.level,
        items=list(block.items) if block.items is not None else None,
        latex=block.latex,
        html=block.html,
        markdown=block.markdown,
        image_path=block.image_path,
        caption=block.caption,
        confidence=block.confidence,
        metadata=dict(block.metadata or {}),
        id=block.id,
    )
