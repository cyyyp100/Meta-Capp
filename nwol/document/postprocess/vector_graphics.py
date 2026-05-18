from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from document.models import BoundingBox, DocumentBlock
from document.postprocess.figure_extractor import document_asset_dir

logger = logging.getLogger("Document.vector_graphics")

_AXIS_WORD_RE = re.compile(
    r"\b(?:x|y|n|valeur|time|temps|emissions?|émissions?|epoch|epochs?|"
    r"accuracy|validation|dice|loss|score|auc|precision|recall|iou|dsc)\b",
    re.I,
)
_AXIS_LABEL_WORD_RE = re.compile(
    r"^(?:x|y|n|valeur|time|temps|emissions?|émissions?|epoch|epochs?|"
    r"accuracy|validation|dice|loss|score|auc|precision|recall|iou|dsc|"
    r"train|training|test|testing|mean|median|error|rate|steps?)$",
    re.I,
)
_NUMERIC_LABEL_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?![A-Za-z])")
_PROSE_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{4,}\b")


def crop_vector_graphic_label_clusters(
    pdf_path: str,
    blocks: list[DocumentBlock],
    output_dir: str | Path | None = None,
    *,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        logger.debug("PyMuPDF indisponible pour crops graphiques vectoriels: %s", exc)
        return blocks

    path = Path(pdf_path)
    out = Path(output_dir) if output_dir is not None else document_asset_dir(path) / "graphics"
    out.mkdir(parents=True, exist_ok=True)

    removed_ids: set[int] = set()
    figures: list[DocumentBlock] = []
    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc):
                page_number = page_index + 1
                if pages is not None and page_number not in pages:
                    continue
                page_blocks = [block for block in blocks if int(block.page or 0) == page_number]
                for group_index, drawing_rect in enumerate(_graph_drawing_rects(page)):
                    nearby_rect = fitz.Rect(
                        drawing_rect.x0 - 96.0,
                        drawing_rect.y0 - 64.0,
                        drawing_rect.x1 + 96.0,
                        drawing_rect.y1 + 64.0,
                    )
                    labels = [
                        block
                        for block in page_blocks
                        if block.bbox is not None
                        and _is_graphic_label_candidate(block)
                        and _rects_touch(fitz.Rect(*block.bbox.to_list()), nearby_rect)
                    ]
                    nearby_captions = [
                        block
                        for block in page_blocks
                        if block.bbox is not None
                        and _is_caption_block(block)
                        and _horizontally_related_rects(fitz.Rect(*block.bbox.to_list()), nearby_rect)
                    ]
                    internal_labels = [
                        block
                        for block in page_blocks
                        if block.bbox is not None
                        and _is_text_inside_graphic(block, drawing_rect)
                    ]
                    semantic_diagram_text = [
                        block
                        for block in page_blocks
                        if block.bbox is not None
                        and _is_semantic_diagram_text_block(block, drawing_rect)
                    ]
                    # Also remove table blocks whose center falls inside the graphic region
                    # (e.g. attention heatmap cells extracted as table artifacts).
                    table_artifacts = [
                        block
                        for block in page_blocks
                        if block.type == "table"
                        and block.bbox is not None
                        and _bbox_center_inside_rect(block.bbox, drawing_rect, margin=20.0)
                    ]
                    associated_text_blocks = list(
                        {id(block): block for block in [*labels, *internal_labels, *semantic_diagram_text, *table_artifacts]}.values()
                    )
                    label_bbox = _union_bbox(labels)
                    rect = drawing_rect
                    if label_bbox is not None:
                        rect = rect | fitz.Rect(*label_bbox.to_list())
                    raw_axis_label_bbox = _raw_axis_label_bbox(page, drawing_rect)
                    if raw_axis_label_bbox is not None:
                        rect = rect | raw_axis_label_bbox
                    padding = 12.0 if _looks_like_compact_flow_rect(drawing_rect) else 28.0
                    rect = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)
                    rect = _exclude_nearby_non_graphic_text(rect, drawing_rect, page_blocks, associated_text_blocks)
                    if raw_axis_label_bbox is not None:
                        rect = _exclude_prose_above_embedded_axis_label(rect, page_blocks, raw_axis_label_bbox)
                    rect = _exclude_nearby_captions(rect, drawing_rect, nearby_captions)
                    rect = _exclude_raw_captions(rect, drawing_rect, page)
                    rect = fitz.Rect(
                        max(rect.x0, page.rect.x0),
                        max(rect.y0, page.rect.y0),
                        min(rect.x1, page.rect.x1),
                        min(rect.y1, page.rect.y1),
                    )
                    if rect.is_empty or rect.width < 90 or rect.height < 40:
                        continue
                    digest = hashlib.md5(f"{path}-{page_number}-{rect}".encode()).hexdigest()[:12]
                    image_path = out / f"graphic_p{page_number}_{group_index}_{digest}.png"
                    if not image_path.exists():
                        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3, 3), alpha=False)
                        pix.save(str(image_path))

                    _strip_embedded_diagram_suffixes(page_blocks, rect)
                    for block in associated_text_blocks:
                        removed_ids.add(id(block))
                    figures.append(
                        DocumentBlock(
                            type="figure",
                            text="",
                            page=page_number,
                            bbox=BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                            image_path=str(image_path),
                            caption="",
                            confidence=0.72,
                            metadata={
                                "source": "vector_graphic_drawing",
                                "contains_schema": True,
                                "caption_display": False,
                                "llm_assets": [{"type": "image", "path": str(image_path), "reason": "vector_graphic"}],
                            },
                        )
                    )
    except Exception as exc:
        logger.warning("Crop des graphiques vectoriels échoué: %s", exc)
        return blocks

    if not figures:
        return blocks
    kept = [
        block
        for block in blocks
        if id(block) not in removed_ids and not _is_residual_text_inside_vector_figure(block, figures)
    ]
    kept.extend(figures)
    return sorted(kept, key=_position_key)


def _is_residual_text_inside_vector_figure(block: DocumentBlock, figures: list[DocumentBlock]) -> bool:
    if block.bbox is None or block.page is None or _is_caption_block(block):
        return False
    if block.type not in {"paragraph", "text", "formula", "heading", "subheading", "subsubheading", "table"}:
        return False
    if block.type in {"heading", "subheading", "subsubheading"} and _looks_like_real_heading(block.text or ""):
        return False

    import fitz  # type: ignore

    for figure in figures:
        if figure.bbox is None or figure.page != block.page:
            continue
        rect = fitz.Rect(*figure.bbox.to_list())
        if not _bbox_center_inside_rect(block.bbox, rect, margin=4.0):
            continue
        if block.type == "table":
            return True
        if _is_text_inside_graphic(block, rect) or _is_graphic_label_candidate(block):
            return True
    return False


def _graph_drawing_rects(page: object) -> list[object]:
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    rects = []
    compact_shape_rects = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None or rect.is_empty:
            continue
        item_count = len(drawing.get("items", []) or [])
        drawing_type = str(drawing.get("type") or "")
        has_stroke = "s" in drawing_type
        has_fill = "f" in drawing_type
        if _looks_like_text_panel(rect, has_fill=has_fill, has_stroke=has_stroke, page=page):
            continue
        if has_stroke and item_count >= 20 and rect.width >= 120.0 and 55.0 <= rect.height <= 260.0:
            rects.append(rect)
            continue
        if has_stroke and item_count >= 5 and rect.width >= 80.0 and 30.0 <= rect.height <= 400.0:
            rects.append(rect)
            continue
        aspect = rect.width / max(rect.height, 1.0)
        if (
            has_fill
            and not has_stroke
            and item_count <= 3
            and rect.width >= 70.0
            and 40.0 <= rect.height <= 260.0
            and 0.35 <= aspect <= 5.0
        ):
            rects.append(rect)
            continue
        if has_fill and has_stroke and item_count >= 3 and rect.width >= 45.0 and 12.0 <= rect.height <= 90.0:
            compact_shape_rects.append(rect)

    if len(compact_shape_rects) >= 3:
        rects.extend(
            rect
            for rect, count in _merge_rects_with_counts(compact_shape_rects, tolerance=128.0)
            if count >= 3
        )

    # Detect dense clusters of small fill-only rectangles (attention heatmaps, confusion matrices).
    # These are rows of filled cells with no stroke — invisible to the stroke-based logic above.
    fill_cell_rects = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None or rect.is_empty:
            continue
        drawing_type = str(drawing.get("type") or "")
        if "f" not in drawing_type or "s" in drawing_type:
            continue
        item_count = len(drawing.get("items", []) or [])
        if item_count <= 3 and 3.0 <= rect.width <= 50.0 and 3.0 <= rect.height <= 80.0:
            fill_cell_rects.append(rect)

    if len(fill_cell_rects) >= 15:
        # Compute the bounding box of all fill cells as a single heatmap region.
        # Using union instead of cluster-merge avoids splitting a single heatmap
        # whose rows have a vertical gap (e.g. separated by text label rows).
        heatmap_rect = fill_cell_rects[0]
        for r in fill_cell_rects[1:]:
            heatmap_rect = heatmap_rect | r
        if heatmap_rect.width >= 60.0 and heatmap_rect.height >= 40.0:
            rects.append(heatmap_rect)

    return _merge_rects(rects)


def _looks_like_text_panel(rect: object, has_fill: bool, has_stroke: bool, page: object) -> bool:
    if not has_fill or has_stroke:
        return False
    aspect = rect.width / max(rect.height, 1.0)
    try:
        page_width = float(page.rect.width)
    except Exception:
        page_width = 0.0
    return rect.width >= 220.0 and (aspect >= 2.2 or (page_width > 0.0 and rect.width >= page_width * 0.45))


def _is_graphic_label_candidate(block: DocumentBlock) -> bool:
    if block.page is None or block.bbox is None:
        return False
    if _is_caption_block(block):
        return False
    if block.type not in {"paragraph", "formula", "heading", "subheading", "subsubheading"}:
        return False
    if block.type in {"heading", "subheading", "subsubheading"} and _looks_like_real_heading(block.text or ""):
        return False
    if block.type == "formula":
        metadata = block.metadata or {}
        if (metadata.get("formula_mode") == "display" or block.image_path) and not _looks_like_graph_formula_label(block):
            return False
    text = _plain_text(block)
    if not text or len(text) > 120:
        return False
    if block.type == "formula" and _looks_like_graph_formula_label(block):
        return True
    prose_words = [word for word in _PROSE_WORD_RE.findall(text) if word.lower() not in {"valeur"}]
    if _looks_like_axis_label_phrase(text):
        return True
    if _looks_like_internal_diagram_label(text):
        return True
    if len(prose_words) >= 2:
        return False
    return bool(_AXIS_WORD_RE.search(text) or _NUMERIC_LABEL_RE.search(text) or block.type == "formula")


def _looks_like_graph_formula_label(block: DocumentBlock) -> bool:
    if block.bbox is None:
        return False
    text = _plain_text(block)
    if not text or len(text) > 90:
        return False
    if block.bbox.width > 180.0 or block.bbox.height > 42.0:
        return False
    if _looks_like_internal_diagram_label(text):
        return True
    words = _PROSE_WORD_RE.findall(text)
    if len(words) > 8:
        return False
    return bool(
        _AXIS_WORD_RE.search(text)
        or _NUMERIC_LABEL_RE.search(text)
        or re.search(r"\b(?:seed|maml|train|test|loss|acc|accuracy)\b", text, re.I)
    )


def _is_caption_block(block: DocumentBlock) -> bool:
    if (block.metadata or {}).get("is_caption"):
        return True
    return bool(re.match(r"^\s*(?:figure|fig\.?|schema|schéma|graphique|diagramme|tableau|table)\b", block.text or "", re.I))


def _is_semantic_diagram_text_block(block: DocumentBlock, drawing_rect: object) -> bool:
    if block.type not in {"paragraph", "text", "abstract"} or block.bbox is None:
        return False
    metadata = block.metadata or {}
    if not (metadata.get("semantic_only_block") or metadata.get("source") == "opendataloader_pdf"):
        return False
    text = _plain_text(block)
    if not re.search(r"\b(?:figure|fig\.?|schema|schéma|graphique|diagramme)\s+\d", text, re.I):
        return False
    if len(text) > 260:
        return False
    import fitz  # type: ignore

    bbox = fitz.Rect(*block.bbox.to_list())
    return _rects_touch(bbox, _expanded_rect(drawing_rect, 36.0))


def _exclude_nearby_captions(rect: object, drawing_rect: object, captions: list[DocumentBlock]) -> object:
    if not captions:
        return rect
    import fitz  # type: ignore

    adjusted = fitz.Rect(rect)
    for caption in captions:
        if caption.bbox is None:
            continue
        cap = fitz.Rect(*caption.bbox.to_list())
        if cap.y0 >= drawing_rect.y1 and cap.y0 - drawing_rect.y1 <= 90.0:
            adjusted.y1 = min(adjusted.y1, max(adjusted.y0, cap.y0 - 4.0))
        elif drawing_rect.y0 >= cap.y1 and drawing_rect.y0 - cap.y1 <= 90.0:
            adjusted.y0 = max(adjusted.y0, min(adjusted.y1, cap.y1 + 4.0))
    return adjusted


def _exclude_raw_captions(rect: object, drawing_rect: object, page: object) -> object:
    import fitz  # type: ignore

    adjusted = fitz.Rect(rect)
    for cap in _raw_caption_rects(page):
        if not _horizontally_related_rects(cap, drawing_rect):
            continue
        if cap.y0 >= drawing_rect.y1 and cap.y0 - drawing_rect.y1 <= 110.0:
            adjusted.y1 = min(adjusted.y1, max(adjusted.y0, cap.y0 - 4.0))
        elif drawing_rect.y0 >= cap.y1 and drawing_rect.y0 - cap.y1 <= 110.0:
            adjusted.y0 = max(adjusted.y0, min(adjusted.y1, cap.y1 + 4.0))
    return adjusted


def _raw_caption_rects(page: object) -> list[object]:
    import fitz  # type: ignore

    rects: list[object] = []
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return rects
    for block in text_dict.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            spans = line.get("spans", []) or []
            text = re.sub(r"\s+", " ", "".join(str(span.get("text") or "") for span in spans)).strip()
            if not re.match(r"^\s*(?:figure|fig\.?|schema|schéma|graphique|diagramme|tableau|table)\b", text, re.I):
                continue
            bbox = line.get("bbox") or block.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                rects.append(fitz.Rect(*(float(value) for value in bbox[:4])))
    return rects


def _horizontally_related_rects(left: object, right: object) -> bool:
    overlap = min(left.x1, right.x1) - max(left.x0, right.x0)
    if overlap > 0:
        return True
    return abs((left.x0 + left.x1) / 2.0 - (right.x0 + right.x1) / 2.0) <= max(left.width, right.width) * 0.65


def _is_text_inside_graphic(block: DocumentBlock, rect: object) -> bool:
    if block.page is None or block.bbox is None or block.type not in {"paragraph", "formula", "heading", "subheading", "subsubheading"}:
        return False
    if block.type in {"heading", "subheading", "subsubheading"} and _looks_like_real_heading(block.text or ""):
        return False
    if block.type == "formula":
        metadata = block.metadata or {}
        if metadata.get("formula_mode") == "display" or block.image_path:
            return False
    text = _plain_text(block)
    if not text or len(text) > 180:
        return False
    bbox = block.bbox
    prose_words = [word for word in _PROSE_WORD_RE.findall(text) if word.lower() not in {"valeur"}]
    if block.type != "formula":
        if len(prose_words) > 6 and not _looks_like_internal_diagram_label(text):
            return False
        rect_width = float(getattr(rect, "width", rect.x1 - rect.x0))
        if bbox.width > max(240.0, rect_width * 0.55):
            return False
    center_x = bbox.center_x
    center_y = bbox.center_y
    return (
        rect.x0 - 8.0 <= center_x <= rect.x1 + 8.0
        and rect.y0 - 8.0 <= center_y <= rect.y1 + 8.0
    )


def _exclude_nearby_non_graphic_text(
    rect: object,
    drawing_rect: object,
    page_blocks: list[DocumentBlock],
    associated_text_blocks: list[DocumentBlock],
) -> object:
    import fitz  # type: ignore

    associated_ids = {id(block) for block in associated_text_blocks}
    adjusted = fitz.Rect(rect)
    for block in page_blocks:
        if id(block) in associated_ids or block.bbox is None or _is_caption_block(block):
            continue
        if block.type not in {"paragraph", "text", "abstract"}:
            continue
        text = _plain_text(block)
        if len(_PROSE_WORD_RE.findall(text)) < 3:
            continue
        if _strip_embedded_diagram_suffix(text) != text:
            continue
        bbox = fitz.Rect(*block.bbox.to_list())
        horizontal_overlap = min(adjusted.x1, bbox.x1) - max(adjusted.x0, bbox.x0)
        if horizontal_overlap <= min(adjusted.width, bbox.width) * 0.12:
            continue
        touches_top_band = bbox.y0 <= drawing_rect.y0 + 16.0 and bbox.y1 <= drawing_rect.y0 + 48.0
        if (bbox.y0 < drawing_rect.y0 or touches_top_band) and bbox.y1 > adjusted.y0:
            adjusted.y0 = max(adjusted.y0, min(adjusted.y1, bbox.y1 + 4.0))
    return adjusted


def _exclude_prose_above_embedded_axis_label(
    rect: object,
    page_blocks: list[DocumentBlock],
    raw_axis_label_bbox: object,
) -> object:
    """When graph labels were glued to prose, keep the labels but drop prose above."""
    import fitz  # type: ignore

    adjusted = fitz.Rect(rect)
    for block in page_blocks:
        if block.bbox is None or block.type not in {"paragraph", "text", "abstract"}:
            continue
        text = _plain_text(block)
        if _strip_embedded_diagram_suffix(text) == text:
            continue
        bbox = fitz.Rect(*block.bbox.to_list())
        if bbox.y0 < raw_axis_label_bbox.y0 < bbox.y1:
            adjusted.y0 = max(adjusted.y0, min(adjusted.y1, raw_axis_label_bbox.y0 - 8.0))
    return adjusted


def _strip_embedded_diagram_suffixes(page_blocks: list[DocumentBlock], drawing_rect: object) -> None:
    for block in page_blocks:
        if block.bbox is None or block.type not in {"paragraph", "text", "abstract"}:
            continue
        if not block.text or len(block.text) < 80:
            continue
        bbox = block.bbox
        if bbox.y1 < drawing_rect.y0 - 8.0 or bbox.y0 > drawing_rect.y1:
            continue
        stripped = _strip_embedded_diagram_suffix(block.text)
        if stripped != block.text:
            block.text = stripped


def _strip_embedded_diagram_suffix(text: str) -> str:
    """Remove a short diagram label accidentally glued after prose text."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    match = re.search(r"(.+[.!?])\s+([A-Z][A-Za-zÀ-ÿ]*(?:\s+[A-Z][A-Za-zÀ-ÿ]*){0,3})$", normalized)
    if not match:
        match = re.search(r"(.+[.!?])\s+([A-Z][A-Za-zÀ-ÿ]*(?:\s+[A-Z][A-Za-zÀ-ÿ]*){0,9})$", normalized)
        if not match:
            match = re.search(r"(.+[.!?])\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]*(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]*){0,9})$", normalized)
            if not match:
                return text
    suffix = match.group(2).strip()
    if not _looks_like_diagram_label_suffix(suffix):
        return text
    return match.group(1).strip()


def _looks_like_diagram_label_suffix(suffix: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]*", suffix or "")
    if not 1 <= len(words) <= 10:
        return False
    if any(len(word) < 3 for word in words):
        return False
    label_hits = sum(1 for word in words if _DIAGRAM_LABEL_WORD_RE.search(word))
    if label_hits == len(words):
        return True
    if len(words) <= 4 and label_hits >= max(1, len(words) - 1):
        return True
    titlecase_hits = sum(1 for word in words if word[:1].isupper())
    if len(words) <= 4:
        return titlecase_hits == len(words)
    return titlecase_hits == len(words) and label_hits >= 2


def _looks_like_real_heading(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped:
        return False
    return bool(re.match(r"^(?:\d+(?:\.\d+)*\.?|chapter|chapitre|part|section)\b", stripped, re.I))


def _looks_like_compact_flow_rect(rect: object) -> bool:
    aspect = rect.width / max(rect.height, 1.0)
    return rect.width >= 180.0 and 18.0 <= rect.height <= 120.0 and aspect <= 4.5


def _looks_like_graph_group(group: list[DocumentBlock]) -> bool:
    if len(group) < 4:
        return False
    bbox = _union_bbox(group)
    if bbox is None or bbox.width < 60 or bbox.height < 45:
        return False
    text = " ".join(_plain_text(block) for block in group)
    axis_hits = len(_AXIS_WORD_RE.findall(text))
    numeric_hits = len(_NUMERIC_LABEL_RE.findall(text))
    return axis_hits >= 1 and numeric_hits >= 2


def _looks_like_axis_label_phrase(text: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text or "")
    if not 1 <= len(words) <= 5:
        return False
    return all(_AXIS_LABEL_WORD_RE.match(word) for word in words)


def _raw_axis_label_bbox(page: object, drawing_rect: object) -> object | None:
    """Find axis labels that the block classifier may have merged or typed as math."""
    import fitz  # type: ignore

    nearby = _expanded_rect(drawing_rect, 120.0)
    rects: list[object] = []
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return None

    for block in text_dict.get("blocks", []) or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            spans = line.get("spans", []) or []
            text = re.sub(r"\s+", " ", "".join(str(span.get("text") or "") for span in spans)).strip()
            if not text or len(text) > 90:
                continue
            bbox = line.get("bbox") or block.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            rect = fitz.Rect(*(float(value) for value in bbox[:4]))
            if not _rects_touch(rect, nearby):
                continue
            if _looks_like_axis_label_phrase(text) or _looks_like_numeric_tick_label(text):
                rects.append(rect)

    if not rects:
        return None
    result = fitz.Rect(rects[0])
    for rect in rects[1:]:
        result |= rect
    return result


def _looks_like_numeric_tick_label(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", re.sub(r"\s+", "", text or "")))


_DIAGRAM_LABEL_WORD_RE = re.compile(
    r"\b(?:adapter|attention|backbone|decoder|encoder|flexible|frozen|glue|image|"
    r"layer|learnable|mask|meta|module|optimizer|prompt|sampler|sampling|self|"
    r"transformer|tune|culture|diplomacy|education|european|power|soft|values|"
    r"carbon|consumption|decarbonization|desired|emissions?|intensity|path|"
    r"population|time|trajectory)\b",
    re.I,
)


def _looks_like_internal_diagram_label(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped or len(stripped) > 140:
        return False
    if re.search(r"[.!?]\s*$", stripped):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", stripped)
    if len(words) > 14:
        return False
    label_hits = len(_DIAGRAM_LABEL_WORD_RE.findall(stripped))
    if len(words) < 3:
        return label_hits >= 1
    if label_hits >= max(3, len(words) // 2):
        return True
    unique_ratio = len({word.casefold() for word in words}) / max(len(words), 1)
    return unique_ratio <= 0.55


def _merge_rects(rects: list[object], tolerance: float = 24.0) -> list[object]:
    return [rect for rect, _count in _merge_rects_with_counts(rects, tolerance=tolerance)]


def _merge_rects_with_counts(rects: list[object], tolerance: float) -> list[tuple[object, int]]:
    merged: list[tuple[object, int]] = []
    for rect in sorted(rects, key=lambda item: (item.y0, item.x0)):
        matched_index = None
        for index, (existing, _count) in enumerate(merged):
            if _expanded_rect(existing, tolerance).intersects(rect):
                matched_index = index
                break
        if matched_index is None:
            merged.append((rect, 1))
        else:
            existing, count = merged[matched_index]
            merged[matched_index] = (existing | rect, count + 1)
    return _consolidate_merged_rects(merged, tolerance)


def _consolidate_merged_rects(merged: list[tuple[object, int]], tolerance: float) -> list[tuple[object, int]]:
    changed = True
    while changed:
        changed = False
        result: list[tuple[object, int]] = []
        for rect, count in merged:
            matched_index = None
            for index, (existing, _existing_count) in enumerate(result):
                if _expanded_rect(existing, tolerance).intersects(rect):
                    matched_index = index
                    break
            if matched_index is None:
                result.append((rect, count))
                continue
            existing, existing_count = result[matched_index]
            result[matched_index] = (existing | rect, existing_count + count)
            changed = True
        merged = result
    return merged


def _expanded_rect(rect: object, amount: float) -> object:
    import fitz  # type: ignore

    return fitz.Rect(rect.x0 - amount, rect.y0 - amount, rect.x1 + amount, rect.y1 + amount)


def _rects_touch(left: object, right: object) -> bool:
    return left.intersects(right) or right.contains(left) or left.contains(right)


def _plain_text(block: DocumentBlock) -> str:
    text = block.latex or block.text or ""
    text = text.replace("$", " ")
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _union_bbox(blocks: list[DocumentBlock]) -> BoundingBox | None:
    bbox = blocks[0].bbox if blocks else None
    for block in blocks[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    return bbox


def _bbox_center_inside_rect(bbox: BoundingBox, rect: object, margin: float = 0.0) -> bool:
    center_x = (bbox.x0 + bbox.x1) / 2.0
    center_y = (bbox.y0 + bbox.y1) / 2.0
    return (
        rect.x0 - margin <= center_x <= rect.x1 + margin
        and rect.y0 - margin <= center_y <= rect.y1 + margin
    )


def _position_key(block: DocumentBlock) -> tuple[int, float, float]:
    if block.bbox is None:
        return (int(block.page or 0), float("inf"), float("inf"))
    return (int(block.page or 0), block.bbox.y0, block.bbox.x0)
