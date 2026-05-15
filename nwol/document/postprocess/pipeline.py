from __future__ import annotations

from document.layout.reading_order import order_blocks_for_reading
from document.models import DocumentBlock
from document.postprocess.context_assets import crop_complex_context_blocks
from document.postprocess.algorithm_figures import crop_algorithm_blocks
from document.postprocess.figure_extractor import associate_captions, deduplicate_visual_blocks, extract_native_figures
from document.postprocess.formula_cropper import crop_formula_blocks
from document.postprocess.inline_formula_repair import repair_fragmented_inline_formulas
from document.postprocess.learning_normalizer import normalize_for_learning, promote_numbered_paragraph_headings
from document.postprocess.list_normalizer import normalize_lists
from document.postprocess.math_fragments import repair_display_math_fragments
from document.postprocess.math_normalizer import normalize_math_blocks
from document.postprocess.math_paragraph_grouper import group_math_dense_paragraphs_until_heading
from document.postprocess.math_visual_cleanup import cleanup_visual_math_fragments
from document.postprocess.paragraph_rebuilder import rebuild_paragraphs, split_long_paragraph
from document.postprocess.table_normalizer import crop_rule_based_tables, crop_table_blocks, extract_native_tables, normalize_tables
from document.postprocess.vector_graphics import crop_vector_graphic_label_clusters


def postprocess_document_blocks(
    blocks: list[DocumentBlock],
    pdf_path: str | None = None,
    page_sizes: dict[int, tuple[float, float]] | None = None,
    *,
    enrich_assets: bool = True,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    """Run the shared block cleanup/enrichment chain for every PDF backend."""
    blocks = normalize_math_blocks(blocks)
    blocks = repair_fragmented_inline_formulas(blocks)
    blocks = normalize_math_blocks(blocks)
    blocks = normalize_lists(blocks)

    if pdf_path and enrich_assets:
        blocks = extract_native_tables(pdf_path, blocks, pages=pages)
    blocks = normalize_tables(blocks)
    if pdf_path and enrich_assets:
        blocks = crop_table_blocks(pdf_path, blocks, pages=pages)

    if page_sizes:
        blocks = order_blocks_for_reading(blocks, page_sizes)  # type: ignore[arg-type]
    blocks = promote_numbered_paragraph_headings(blocks)
    if pdf_path and enrich_assets:
        blocks = crop_algorithm_blocks(pdf_path, blocks, pages=pages)
    blocks = rebuild_paragraphs(blocks, page_sizes=page_sizes)
    blocks = normalize_math_blocks(blocks)

    if pdf_path and enrich_assets:
        existing_figures = [block for block in blocks if block.type == "figure"]
        existing_tables = [block for block in blocks if block.type == "table"]
        text_blocks = [block for block in blocks if block.type not in {"figure", "table"}]
        figures = deduplicate_visual_blocks([*existing_figures, *extract_native_figures(pdf_path, pages=pages)])
        blocks = associate_captions(text_blocks, [*existing_tables, *figures])
        if page_sizes:
            blocks = order_blocks_for_reading(blocks, page_sizes)  # type: ignore[arg-type]

    blocks = repair_display_math_fragments(blocks)
    blocks = repair_fragmented_inline_formulas(blocks)
    blocks = rebuild_paragraphs(blocks, page_sizes=page_sizes)
    blocks = normalize_math_blocks(blocks)
    blocks = cleanup_visual_math_fragments(blocks)
    blocks = group_math_dense_paragraphs_until_heading(blocks)
    blocks = _split_long_reader_paragraphs(blocks)

    if pdf_path and enrich_assets:
        blocks = crop_formula_blocks(pdf_path, blocks)
        blocks = crop_rule_based_tables(pdf_path, blocks, pages=pages)
        blocks = crop_vector_graphic_label_clusters(pdf_path, blocks, pages=pages)
        blocks = deduplicate_visual_blocks(blocks)
        blocks = crop_complex_context_blocks(pdf_path, blocks)

    blocks = _drop_empty_inert_blocks(blocks)
    if page_sizes:
        blocks = order_blocks_for_reading(blocks, page_sizes)  # type: ignore[arg-type]

    return normalize_for_learning(blocks)


def _split_long_reader_paragraphs(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for block in blocks:
        if (
            block.type == "paragraph"
            and len((block.text or "").strip()) > 900
            and (block.metadata or {}).get("render_mode") != "context_crop_only"
            and (block.metadata or {}).get("reader_render_mode") != "context_crop_only"
        ):
            result.extend(split_long_paragraph(block))
        else:
            result.append(block)
    return result


def _drop_empty_inert_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for block in blocks:
        metadata = block.metadata or {}
        if block.type == "table" and not (
            (block.text or "").strip()
            or (block.markdown or "").strip()
            or (block.html or "").strip()
            or block.image_path
            or metadata.get("table_image_path")
            or metadata.get("context_asset_path")
        ):
            continue
        result.append(block)
    return result
