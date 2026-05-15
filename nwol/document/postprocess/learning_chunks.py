from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from document.models import BoundingBox, DocumentBlock, LearningChunk, VisualAsset

_SCIENTIFIC_SIGNALS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "methods",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
    "bibliography",
    "doi",
    "arxiv",
)
_REFERENCE_HEADING_RE = re.compile(r"^\s*(?:references|bibliography|bibliographie)\s*$", re.I)
_APPENDIX_HEADING_RE = re.compile(
    r"^\s*(?:(?:appendix|annexe|supplementary|supplemental)\b|additional\s+results\b)",
    re.I,
)
_METADATA_RE = re.compile(
    r"(@|doi\s*:|doi\.org|arxiv\s*:|issn\s*:|isbn\s*:|copyright|creative\s+commons|"
    r"\b(?:accepted|submitted|preprint)\b)",
    re.I,
)
_AFFILIATION_RE = re.compile(
    r"\b(?:university|college|institute|department|school|laborator(?:y|ies)|lab|"
    r"faculty|academy|hospital|centre|center|cnrs|inria)\b",
    re.I,
)
_AUTHOR_MARKER_RE = re.compile(r"(?:[*†‡]|\\dagger|\\ddagger|\^\{?\d)")
_VISUAL_MENTION_RE = re.compile(
    r"\b(?:figure|fig\.?|image|schema|schéma|diagram|diagramme|graph|graphique|"
    r"table|tableau|equation|équation|formula|formule)\s*\d*",
    re.I,
)
_LATEX_SIGNAL_RE = re.compile(r"\\[A-Za-z]{2,}|\$[^$\n]{1,200}\$")
_MATH_TEXT_SIGNAL_RE = re.compile(
    r"(?<!\w)[A-Za-z]\s*[_^]\s*[A-Za-z0-9{(]"
    r"|[∑∫√∞≈≠≤≥→←↔∈∉∀∃αβγδλμσφψω]"
    r"|\b(?:lim|sin|cos|tan|ln|log|exp)\b\s*[_({]?"
)
_TEXTUAL_TYPES = {
    "paragraph",
    "text",
    "abstract",
    "definition",
    "theorem",
    "example",
    "remark",
    "warning",
    "exercise",
    "question",
    "quote",
}
_INTERACTIVE_TYPES = _TEXTUAL_TYPES | {"bullet_list"}


def is_geometrically_valid(block: DocumentBlock) -> bool:
    return (
        block.page is not None
        and block.bbox is not None
        and block.bbox.width > 8.0
        and block.bbox.height > 6.0
    )


def detect_document_type(blocks: list[DocumentBlock], pages: int = 0) -> str:
    text = "\n".join(_block_text(block) for block in blocks if _block_text(block)).casefold()
    text_chars = len(re.sub(r"\s+", "", text))
    if pages and text_chars < max(80, pages * 25):
        return "scanned_pdf"

    scientific = detect_scientific_article(blocks)
    two_column_pages = _two_column_page_count(blocks)
    visual_count = sum(1 for block in blocks if _block_visual_assets(block) or block.type in {"figure", "table"})
    formula_count = sum(1 for block in blocks if block.type == "formula" or _block_has_math(block))
    blocks_per_page = len(blocks) / max(1, pages or max((int(block.page or 0) for block in blocks), default=1))

    if scientific and two_column_pages:
        return "scientific_article_two_columns"
    if scientific:
        return "scientific_article"
    if blocks_per_page <= 7 and pages >= 2:
        return "slides"
    if visual_count >= max(3, len(blocks) * 0.18):
        return "mixed_visual_pdf"
    if formula_count >= max(2, len(blocks) * 0.12):
        return "course_math"
    return "course_simple"


def detect_scientific_article(blocks: Iterable[DocumentBlock]) -> bool:
    text = "\n".join(_block_text(block) for block in blocks if _block_text(block)).casefold()
    score = sum(1 for signal in _SCIENTIFIC_SIGNALS if signal in text)
    return score >= 3


def detect_two_column_layout(page_blocks: list[DocumentBlock], page_width: float | None = None) -> bool:
    valid = [block for block in page_blocks if block.bbox is not None]
    if not valid:
        return False
    width = page_width or _page_width(valid)
    if width <= 0:
        return False
    left = [block for block in valid if block.bbox and block.bbox.center_x < width * 0.48]
    right = [block for block in valid if block.bbox and block.bbox.center_x > width * 0.52]
    return len(left) >= 4 and len(right) >= 4


def text_similarity(a: str | None, b: str | None) -> float:
    a_words = set((a or "").casefold().split())
    b_words = set((b or "").casefold().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def enrich_blocks_for_learning(
    blocks: list[DocumentBlock],
    *,
    document_type: str | None = None,
) -> list[DocumentBlock]:
    doc_type = document_type or detect_document_type(blocks)
    _mark_document_metadata(blocks, doc_type)
    _attach_visuals_to_nearest_blocks(blocks)
    for index, block in enumerate(blocks):
        block.metadata.setdefault("document_type", doc_type)
        block.metadata.setdefault("chunk_type", _chunk_type(block))
        block.metadata["quality_score"] = compute_chunk_quality(block)
        block.metadata["generation_mode"] = choose_generation_mode(block)
        block.metadata["render_type"] = _render_type(block)
        mark_displayable(block)
        block.metadata["interactive"] = _is_interactive(block)
        block.metadata.setdefault("source_blocks", [block.id or f"block_{index}"])
    _suppress_column_fragment_duplicates(blocks)
    _suppress_semantic_duplicate_blocks(blocks)
    return blocks


def mark_displayable(block: DocumentBlock) -> DocumentBlock:
    text = _block_text(block).strip()
    metadata = block.metadata or {}
    displayable = bool(
        is_geometrically_valid(block)
        and len(text) >= 40
        and not metadata.get("is_metadata")
        and not metadata.get("is_header_footer")
        and not metadata.get("is_reference")
        and not _is_isolated_caption(block)
        and not _is_isolated_formula(block)
        and block.type in _INTERACTIVE_TYPES
    )
    block.metadata["displayable"] = displayable
    return block


def compute_chunk_quality(block: DocumentBlock) -> float:
    metadata = block.metadata or {}
    text = _block_text(block).strip()
    score = float(block.confidence if block.confidence is not None else 1.0)

    if not is_geometrically_valid(block):
        score -= 0.25
    if not text or len(text) < 60:
        score -= 0.25
    if metadata.get("is_reference"):
        score -= 0.50
    if metadata.get("is_header_footer"):
        score -= 0.40
    if metadata.get("mixed_columns_risk"):
        score -= 0.20
    if metadata.get("broken_math_risk"):
        score -= 0.20
    if metadata.get("has_unattached_visual"):
        score -= 0.15
    if _block_visual_assets(block) and not text:
        score -= 0.15
    if _is_isolated_caption(block):
        score -= 0.35
    if _is_isolated_formula(block):
        score -= 0.20

    return round(max(0.0, min(1.0, score)), 3)


def choose_generation_mode(block: DocumentBlock) -> str:
    metadata = block.metadata or {}
    text = _block_text(block).strip()
    if metadata.get("is_reference") or metadata.get("is_metadata") or metadata.get("is_header_footer"):
        return "classic"
    if block.type == "heading" or _is_isolated_caption(block):
        return "classic"

    quality_score = float(metadata.get("quality_score", block.confidence or 1.0) or 0.0)
    chunk_type = metadata.get("chunk_type") or _chunk_type(block)
    has_latex = bool(block.latex) or metadata.get("contains_inline_math") or _block_has_math(block)
    has_visual_assets = bool(metadata.get("visual_assets")) or bool(metadata.get("llm_assets"))
    has_image = bool(block.image_path)
    has_table = chunk_type == "table" or metadata.get("contains_table")
    has_graph = metadata.get("contains_graph") or metadata.get("contains_schema")
    low_confidence = quality_score < 0.75 and (len(text) >= 60 or has_visual_assets or has_image or has_table)
    crop_only = metadata.get("render_mode") == "context_crop_only"

    if crop_only:
        return "llm_multimodal"
    if has_visual_assets or has_image or has_table or has_graph:
        return "llm_multimodal"
    if has_latex and metadata.get("formula_mode") in {"ambiguous", "display", "complex"}:
        return "llm_text_or_multimodal"
    if low_confidence:
        return "llm_multimodal"
    return "classic"


def build_learning_chunks(blocks: list[DocumentBlock]) -> list[LearningChunk]:
    chunks: list[LearningChunk] = []
    for index, block in enumerate(blocks):
        metadata = block.metadata or {}
        if (
            metadata.get("is_reference")
            or metadata.get("is_metadata")
            or metadata.get("is_header_footer")
            or metadata.get("quality_score", 0.0) < 0.50
        ):
            continue
        text = _block_text(block).strip()
        if not text and not _block_visual_assets(block):
            continue
        page = int(block.page or 1)
        chunks.append(
            LearningChunk(
                id=block.id or f"chunk_{index}",
                title=metadata.get("parent_title") if isinstance(metadata.get("parent_title"), str) else None,
                text=text,
                page_start=page,
                page_end=page,
                bbox=block.bbox,
                source_blocks=list(metadata.get("source_blocks") or [block.id or f"block_{index}"]),
                chunk_type=str(metadata.get("chunk_type") or _chunk_type(block)),
                quality_score=float(metadata.get("quality_score", block.confidence or 0.0) or 0.0),
                generation_mode=str(metadata.get("generation_mode") or "classic"),
                latex=block.latex,
                visual_assets=[_visual_asset_from_dict(item) for item in metadata.get("visual_assets") or []],
                metadata=dict(metadata),
            )
        )
    return chunks


def _mark_document_metadata(blocks: list[DocumentBlock], document_type: str) -> None:
    in_references = False
    after_reference_appendix = False
    current_heading = None
    previous_caption: DocumentBlock | None = None
    for block in blocks:
        text = re.sub(r"\s+", " ", _block_text(block)).strip()
        if block.type == "heading":
            if _REFERENCE_HEADING_RE.match(text):
                in_references = True
                after_reference_appendix = False
                block.metadata["is_reference"] = True
            elif in_references and _APPENDIX_HEADING_RE.match(text):
                in_references = False
                after_reference_appendix = True
                block.metadata.pop("is_reference", None)
            if not in_references:
                current_heading = text or current_heading
        elif in_references:
            block.metadata["is_reference"] = True
        elif after_reference_appendix:
            block.metadata.pop("is_reference", None)

        if current_heading and block.type != "heading":
            block.metadata.setdefault("parent_title", current_heading)

        if _METADATA_RE.search(text) or _looks_like_front_matter_affiliation(block, text):
            block.metadata["is_metadata"] = True
        if _continues_previous_caption(block, previous_caption, text):
            block.metadata["is_caption"] = True
        if block.metadata.get("is_caption") and block.type == "paragraph":
            block.metadata.setdefault("caption_isolated", True)
            previous_caption = block
        elif text:
            previous_caption = None
        if document_type == "scientific_article_two_columns" and _looks_like_mixed_column_risk(block):
            block.metadata.setdefault("mixed_columns_risk", True)


def _attach_visuals_to_nearest_blocks(blocks: list[DocumentBlock]) -> None:
    candidates_by_page: dict[int, list[DocumentBlock]] = defaultdict(list)
    for block in blocks:
        if block.type in _TEXTUAL_TYPES and is_geometrically_valid(block):
            if not block.metadata.get("is_metadata") and not block.metadata.get("is_reference"):
                candidates_by_page[int(block.page or 0)].append(block)

    for visual_block in blocks:
        assets = _block_visual_assets(visual_block)
        if not assets:
            continue
        visual_block.metadata.setdefault("visual_assets", [asset.to_dict() for asset in assets])
        visual_block.metadata.setdefault("generation_mode", "llm_multimodal")
        if _is_standalone_visual_block(visual_block):
            continue
        target = _nearest_text_block(visual_block, candidates_by_page.get(int(visual_block.page or 0), []))
        if target is None:
            visual_block.metadata["has_unattached_visual"] = True
            continue
        _attach_assets(target, assets, source_block=visual_block)
        visual_block.metadata["attached_to_block_id"] = target.id


def _looks_like_front_matter_affiliation(block: DocumentBlock, text: str) -> bool:
    if block.page is not None and int(block.page or 0) > 2:
        return False
    if not text:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ0-9'’.-]+", text)
    if len(words) > 32:
        return False
    if _AUTHOR_MARKER_RE.search(text) and _looks_name_or_affiliation_like(text):
        return True
    return bool(_AFFILIATION_RE.search(text) and _looks_like_affiliation_line(text))


def _continues_previous_caption(block: DocumentBlock, previous: DocumentBlock | None, text: str) -> bool:
    if previous is None or block.type != "paragraph" or not text:
        return False
    if block.page != previous.page or block.bbox is None or previous.bbox is None:
        return False
    if block.metadata.get("is_caption") or block.metadata.get("is_reference"):
        return False
    if not re.match(r"^[a-zà-ÿ0-9(]", text):
        return False
    vertical_gap = block.bbox.y0 - previous.bbox.y1
    if vertical_gap > 6.0:
        return False
    horizontal_overlap = min(block.bbox.x1, previous.bbox.x1) - max(block.bbox.x0, previous.bbox.x0)
    return horizontal_overlap >= min(block.bbox.width, previous.bbox.width) * 0.45


def _looks_name_or_affiliation_like(text: str) -> bool:
    proper_words = re.findall(r"\b[A-Z][A-Za-zÀ-ÿ'’.-]{1,}\b", text)
    return len(proper_words) >= 2 or bool(_AFFILIATION_RE.search(text))


def _looks_like_affiliation_line(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"[.!?;:]\s*$", stripped):
        return False
    lowered = stripped.casefold()
    return not re.search(r"\b(?:we|this|these|our|method|model|result|results|figure|table)\b", lowered)


def _nearest_text_block(visual: DocumentBlock, candidates: list[DocumentBlock]) -> DocumentBlock | None:
    if not candidates or visual.bbox is None:
        return None
    max_distance = 280.0 if visual.type in {"figure", "table"} else 180.0
    ranked: list[tuple[float, DocumentBlock]] = []
    for candidate in candidates:
        if candidate is visual or candidate.bbox is None:
            continue
        vertical = min(
            abs(candidate.bbox.center_y - visual.bbox.center_y),
            abs(candidate.bbox.y1 - visual.bbox.y0),
            abs(visual.bbox.y1 - candidate.bbox.y0),
        )
        if vertical > max_distance and not _VISUAL_MENTION_RE.search(candidate.text or ""):
            continue
        bonus = 60.0 if _VISUAL_MENTION_RE.search(candidate.text or "") else 0.0
        ranked.append((max(0.0, vertical - bonus), candidate))
    if not ranked:
        return None
    return min(ranked, key=lambda item: item[0])[1]


def _is_standalone_visual_block(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    return bool(
        block.type == "figure"
        and (
            metadata.get("contains_algorithm")
            or metadata.get("source") == "algorithm_text_panel"
        )
    )


def _attach_assets(target: DocumentBlock, assets: list[VisualAsset], *, source_block: DocumentBlock) -> None:
    metadata = target.metadata
    visual_assets = list(metadata.get("visual_assets") or [])
    llm_assets = list(metadata.get("llm_assets") or [])
    source_ids = list(metadata.get("source_blocks") or ([target.id] if target.id else []))

    for asset in assets:
        asset_dict = asset.to_dict()
        if not _contains_asset(visual_assets, asset_dict):
            visual_assets.append(asset_dict)
        llm_entry = {"type": "image", "path": asset.image_path, "reason": asset.asset_type}
        if not _contains_llm_asset(llm_assets, llm_entry):
            llm_assets.append(llm_entry)
        if asset.asset_type == "table":
            metadata["contains_table"] = True
        elif asset.asset_type == "schema":
            metadata["contains_schema"] = True
        elif asset.asset_type == "graph":
            metadata["contains_graph"] = True
        elif asset.asset_type == "formula":
            metadata.setdefault("formula_mode", "display")

    if source_block.id and source_block.id not in source_ids:
        source_ids.append(source_block.id)
    metadata["visual_assets"] = visual_assets
    metadata["llm_assets"] = llm_assets
    metadata["source_blocks"] = source_ids


def _block_visual_assets(block: DocumentBlock) -> list[VisualAsset]:
    assets: list[VisualAsset] = []
    if block.page is None or block.bbox is None:
        return assets

    metadata = block.metadata or {}
    page = int(block.page)
    source = str(metadata.get("source") or metadata.get("engine") or "unknown")

    if block.type == "figure" and block.image_path:
        asset_type = "schema" if metadata.get("contains_schema") else "figure"
        if metadata.get("contains_graph"):
            asset_type = "graph"
        assets.append(VisualAsset(asset_type, block.image_path, page, block.bbox, block.caption or block.text or None, block.confidence, source))

    table_image = metadata.get("table_image_path")
    if block.type == "table" and table_image:
        assets.append(VisualAsset("table", str(table_image), page, block.bbox, block.caption or None, block.confidence, source))

    formula_image = metadata.get("formula_image_path") or (block.image_path if block.type == "formula" else None)
    if block.type == "formula" and formula_image:
        assets.append(VisualAsset("formula", str(formula_image), page, block.bbox, block.caption or None, block.confidence, source))

    context_path = metadata.get("context_asset_path")
    if context_path:
        reason = str(metadata.get("context_asset_reason") or "context_crop")
        asset_type = "formula" if "math" in reason or "formula" in reason else "context_crop"
        assets.append(VisualAsset(asset_type, str(context_path), page, block.bbox, block.caption or None, block.confidence, source))

    return assets


def _visual_asset_from_dict(data: dict) -> VisualAsset:
    bbox = BoundingBox.from_seq(data.get("bbox"))
    return VisualAsset(
        asset_type=str(data.get("asset_type") or "image"),
        image_path=str(data.get("image_path") or ""),
        page=int(data.get("page") or 1),
        bbox=bbox,
        caption=data.get("caption"),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        source=str(data.get("source") or "unknown"),
    )


def _contains_asset(items: list, candidate: dict) -> bool:
    return any(isinstance(item, dict) and item.get("image_path") == candidate.get("image_path") for item in items)


def _contains_llm_asset(items: list, candidate: dict) -> bool:
    return any(isinstance(item, dict) and item.get("path") == candidate.get("path") for item in items)


def _two_column_page_count(blocks: list[DocumentBlock]) -> int:
    by_page: dict[int, list[DocumentBlock]] = defaultdict(list)
    for block in blocks:
        if block.page is not None:
            by_page[int(block.page)].append(block)
    return sum(1 for page_blocks in by_page.values() if detect_two_column_layout(page_blocks))


def _page_width(blocks: list[DocumentBlock]) -> float:
    for block in blocks:
        value = block.metadata.get("page_width") if block.metadata else None
        try:
            if value:
                return float(value)
        except (TypeError, ValueError):
            pass
    return max((block.bbox.x1 for block in blocks if block.bbox), default=0.0)


def _chunk_type(block: DocumentBlock) -> str:
    if block.type in {"table", "figure", "formula"}:
        return block.type
    if block.metadata.get("visual_assets"):
        return "mixed"
    if _block_has_math(block):
        return "math"
    return block.type or "paragraph"


def _render_type(block: DocumentBlock) -> str:
    metadata = block.metadata or {}
    if metadata.get("quality_score", 1.0) < 0.65:
        return "low_confidence"
    if block.type == "table" or metadata.get("contains_table"):
        return "paragraph_table"
    if block.type == "figure" or metadata.get("contains_graph") or metadata.get("contains_schema"):
        return "paragraph_visual"
    if block.type == "formula" or metadata.get("formula_mode") or _block_has_math(block):
        return "paragraph_math"
    if metadata.get("visual_assets"):
        return "paragraph_mixed"
    return "paragraph_text"


def _is_interactive(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    if not metadata.get("displayable"):
        return False
    if metadata.get("quality_score", 0.0) < 0.65:
        return False
    return block.type in _INTERACTIVE_TYPES


def _is_isolated_caption(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    return bool(metadata.get("is_caption") and not metadata.get("visual_assets"))


def _is_isolated_formula(block: DocumentBlock) -> bool:
    if block.type != "formula":
        return False
    text = _block_text(block).strip()
    return len(text) < 24 and not _block_visual_assets(block)


def _looks_like_mixed_column_risk(block: DocumentBlock) -> bool:
    if block.bbox is None or block.type not in {"paragraph", "heading"}:
        return False
    text = _block_text(block)
    return len(text) >= 40 and block.bbox.width > 360 and "\n" not in text


def _block_has_math(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    if metadata.get("contains_inline_math") or metadata.get("formula_mode"):
        return True
    text = _block_text(block)
    return bool(_LATEX_SIGNAL_RE.search(text) or _MATH_TEXT_SIGNAL_RE.search(text))


def _suppress_column_fragment_duplicates(blocks: list[DocumentBlock]) -> None:
    """Mark narrow column-fragment paragraphs as non-displayable after mark_displayable.

    Two signals, applied only on pages that have at least one wide paragraph (i.e., pages
    where two-column fusion produced full-width merged blocks alongside raw column lines):

    1. Text ends with a hyphen on a narrow block → definitive column-break artifact.
    2. Narrow block whose text is a leading fragment of a wider merged block on the same
       page → de-duplication of raw column lines that survived alongside fused paragraphs.
    """
    para_by_page: dict[int, list[DocumentBlock]] = defaultdict(list)
    for block in blocks:
        if block.type == "paragraph" and block.page is not None and block.bbox is not None:
            para_by_page[int(block.page)].append(block)

    for page_blocks in para_by_page.values():
        page_width = max((b.bbox.x1 for b in page_blocks if b.bbox), default=0.0)
        if page_width <= 0:
            continue

        narrow_threshold = page_width * 0.62
        wide_paragraphs = [
            b for b in page_blocks
            if b.bbox
            and b.bbox.width >= narrow_threshold * 1.1
            and not b.metadata.get("is_metadata")
            and not b.metadata.get("is_header_footer")
        ]
        if not wide_paragraphs:
            continue

        for block in page_blocks:
            if not block.metadata.get("displayable"):
                continue
            if block.bbox is None or block.bbox.width >= narrow_threshold:
                continue
            text = (block.text or "").strip()
            if not text:
                continue

            if text.endswith("-"):
                    block.metadata["displayable"] = False
                    block.metadata["interactive"] = False
                    block.metadata["suppressed_column_fragment"] = True
                    continue

            for wide in wide_paragraphs:
                if _text_is_fragment_of(text, (wide.text or "").strip()):
                    block.metadata["displayable"] = False
                    block.metadata["interactive"] = False
                    block.metadata["suppressed_column_fragment"] = True
                    break


def _suppress_semantic_duplicate_blocks(blocks: list[DocumentBlock]) -> None:
    by_page: dict[int, list[DocumentBlock]] = defaultdict(list)
    for block in blocks:
        if block.page is not None and block.type in _INTERACTIVE_TYPES:
            for page in _block_page_numbers(block):
                by_page[page].append(block)

    for page_blocks in by_page.values():
        anchors = [
            block
            for block in page_blocks
            if not block.metadata.get("semantic_only_block")
            and not block.metadata.get("is_metadata")
            and not block.metadata.get("is_reference")
            and _block_text(block).strip()
        ]
        if not anchors:
            continue

        for block in page_blocks:
            if not block.metadata.get("displayable") or not block.metadata.get("semantic_only_block"):
                continue
            text = _block_text(block).strip()
            if len(text) < 60:
                continue
            other_text = " ".join(_block_text(other) for other in anchors if other is not block)
            if _token_coverage(text, other_text) >= 0.74:
                block.metadata["displayable"] = False
                block.metadata["interactive"] = False
                block.metadata["suppressed_semantic_duplicate"] = True


def _text_is_fragment_of(fragment: str, full_text: str) -> bool:
    """True if fragment is a leading fragment of full_text (hyphen-join normalised)."""
    if not fragment or not full_text:
        return False
    frag_norm = re.sub(r"\s+", " ", re.sub(r"-\s*$", "", fragment.rstrip())).strip().casefold()
    full_norm = re.sub(r"\s+", " ", full_text).strip().casefold()
    if len(frag_norm) < 30:
        return False
    check_len = max(20, len(frag_norm) - 8)
    return full_norm.startswith(frag_norm[:check_len])


def _block_page_numbers(block: DocumentBlock) -> list[int]:
    metadata = block.metadata or {}
    try:
        start = int(metadata.get("page_start") or block.page or 0)
    except (TypeError, ValueError):
        start = int(block.page or 0)
    try:
        end = int(metadata.get("page_end") or start)
    except (TypeError, ValueError):
        end = start
    if start <= 0:
        return []
    if end < start:
        end = start
    end = min(end, start + 2)
    return list(range(start, end + 1))


def _token_coverage(candidate: str, reference: str) -> float:
    candidate_tokens = _content_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    reference_tokens = set(_content_tokens(reference))
    if not reference_tokens:
        return 0.0
    covered = sum(1 for token in candidate_tokens if token in reference_tokens)
    return covered / len(candidate_tokens)


def _content_tokens(text: str) -> list[str]:
    normalized = (
        text.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("+ +", "++")
        .replace("•", " ")
    )
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:[-+][A-Za-zÀ-ÿ0-9]+)*", normalized)
        if len(token) > 1
    ]


def _block_text(block: DocumentBlock) -> str:
    if block.type == "bullet_list":
        return " ".join(block.items or [])
    if block.type == "formula":
        return block.latex or block.text or ""
    if block.type == "table":
        return block.text or block.markdown or block.html or ""
    return block.text or block.caption or ""
