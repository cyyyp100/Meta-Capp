from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from document.models import BoundingBox, DocumentBlock
from document.postprocess.figure_extractor import document_asset_dir

logger = logging.getLogger("Document.algorithm_figures")

_ALGORITHM_TITLE_RE = re.compile(r"^\s*Algorithm\s+\d+\b[^\n]*", re.I)
_ALGORITHM_LINE_RE = re.compile(
    r"^\s*(?:Input\s*:|Require\s*:|Output\s*:|\d+\s*:|for\b|end\s+for\b|if\b|else\b|return\b)",
    re.I,
)
_PROSE_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{4,}\b")


def crop_algorithm_blocks(
    pdf_path: str,
    blocks: list[DocumentBlock],
    output_dir: str | Path | None = None,
    *,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    """Convert pseudocode algorithm panels into schema-like figure crops."""
    try:
        import fitz  # type: ignore
    except Exception as exc:
        logger.debug("PyMuPDF indisponible pour crops algorithmes: %s", exc)
        return blocks

    path = Path(pdf_path)
    out = Path(output_dir) if output_dir is not None else document_asset_dir(path) / "algorithms"
    out.mkdir(parents=True, exist_ok=True)

    regions_by_page: dict[int, list[tuple[object, str]]] = {}
    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                if pages is not None and page_index not in pages:
                    continue
                regions = _detect_algorithm_regions(page)
                if not regions:
                    continue
                page_blocks = [block for block in blocks if int(block.page or 0) == page_index]
                expanded = [
                    (_expand_region_with_blocks(rect, page_blocks), title)
                    for rect, title in regions
                ]
                regions_by_page[page_index] = expanded
    except Exception as exc:
        logger.warning("Détection des algorithmes échouée: %s", exc)
        return blocks

    if not regions_by_page:
        return blocks

    kept: list[DocumentBlock] = []
    for block in blocks:
        regions = regions_by_page.get(int(block.page or 0))
        if regions and _block_inside_algorithm_regions(block, [rect for rect, _title in regions]):
            continue
        kept.append(block)

    figures: list[DocumentBlock] = []
    try:
        with fitz.open(path) as doc:
            for page_number, regions in regions_by_page.items():
                page = doc[page_number - 1]
                for index, (rect, title) in enumerate(regions):
                    clip = fitz.Rect(
                        max(rect.x0 - 6.0, page.rect.x0),
                        max(rect.y0 - 6.0, page.rect.y0),
                        min(rect.x1 + 6.0, page.rect.x1),
                        min(rect.y1 + 6.0, page.rect.y1),
                    )
                    if clip.is_empty or clip.width < 120.0 or clip.height < 45.0:
                        continue
                    digest = hashlib.md5(f"{path}-{page_number}-{clip}".encode()).hexdigest()[:12]
                    image_path = out / f"algorithm_p{page_number}_{index}_{digest}.png"
                    if not image_path.exists():
                        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(3, 3), alpha=False)
                        pix.save(str(image_path))
                    figures.append(
                        DocumentBlock(
                            type="figure",
                            text=title,
                            page=page_number,
                            bbox=BoundingBox(float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)),
                            image_path=str(image_path),
                            caption=title,
                            confidence=0.86,
                            metadata={
                                "source": "algorithm_text_panel",
                                "contains_schema": True,
                                "contains_algorithm": True,
                                "caption_display": True,
                                "llm_assets": [{"type": "image", "path": str(image_path), "reason": "algorithm"}],
                            },
                        )
                    )
    except Exception as exc:
        logger.warning("Crop des algorithmes échoué: %s", exc)
        return blocks

    if not figures:
        return blocks
    return sorted([*kept, *figures], key=_position_key)


def _detect_algorithm_regions(page: object) -> list[tuple[object, str]]:
    import fitz  # type: ignore

    text_blocks = []
    for raw in page.get_text("blocks") or []:
        if len(raw) < 5:
            continue
        text = re.sub(r"\s+", " ", str(raw[4] or "")).strip()
        if not text:
            continue
        rect = fitz.Rect(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        text_blocks.append((rect, text))

    regions: list[tuple[object, str]] = []
    for index, (title_rect, text) in enumerate(text_blocks):
        match = _ALGORITHM_TITLE_RE.match(text)
        if not match:
            continue
        title = match.group(0).strip()
        column = _algorithm_column_rect(page, title_rect)
        included = [title_rect]
        last_y = title_rect.y1
        for rect, block_text in text_blocks[index + 1 :]:
            if rect.y0 < title_rect.y0 - 2.0:
                continue
            if not _rect_center_in_column(rect, column):
                continue
            if rect.y0 - last_y > 72.0:
                break
            if not _looks_like_algorithm_text(block_text):
                if rect.y0 - last_y > 18.0 or _looks_like_prose_block(block_text):
                    break
                if rect.y0 > title_rect.y1 + 220.0:
                    break
            included.append(rect)
            last_y = max(last_y, rect.y1)

        if len(included) < 2:
            continue
        region = included[0]
        for rect in included[1:]:
            region = region | rect
        region = fitz.Rect(max(column.x0, region.x0 - 4.0), region.y0, min(column.x1, region.x1 + 4.0), region.y1)
        if region.height >= 38.0:
            regions.append((region, title))
    return regions


def _algorithm_column_rect(page: object, title_rect: object) -> object:
    import fitz  # type: ignore

    page_rect = page.rect
    midpoint = page_rect.x0 + page_rect.width / 2.0
    title_center_x = (title_rect.x0 + title_rect.x1) / 2.0
    if title_center_x <= midpoint:
        return fitz.Rect(page_rect.x0 + 42.0, page_rect.y0, midpoint - 12.0, page_rect.y1)
    return fitz.Rect(midpoint + 12.0, page_rect.y0, page_rect.x1 - 42.0, page_rect.y1)


def _rect_center_in_column(rect: object, column: object) -> bool:
    center_x = (rect.x0 + rect.x1) / 2.0
    return column.x0 - 8.0 <= center_x <= column.x1 + 8.0


def _looks_like_algorithm_text(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return False
    return bool(_ALGORITHM_TITLE_RE.match(clean) or _ALGORITHM_LINE_RE.match(clean))


def _looks_like_prose_block(text: str) -> bool:
    words = _PROSE_WORD_RE.findall(text or "")
    return len(words) >= 8 and bool(re.search(r"[.!?]\s*$", text.strip()))


def _expand_region_with_blocks(rect: object, page_blocks: list[DocumentBlock]) -> object:
    import fitz  # type: ignore

    expanded = fitz.Rect(rect)
    for block in page_blocks:
        if block.bbox is None:
            continue
        text = _block_text(block)
        bbox = fitz.Rect(*block.bbox.to_list())
        center_x = (bbox.x0 + bbox.x1) / 2.0
        if not (expanded.x0 - 10.0 <= center_x <= expanded.x1 + 10.0):
            continue
        if bbox.y0 < expanded.y0 - 4.0 or bbox.y0 > expanded.y1 + 42.0:
            continue
        if _looks_like_algorithm_text(text) or _algorithm_overlap_ratio(bbox, expanded) > 0.18:
            expanded = expanded | bbox
    return expanded


def _block_inside_algorithm_regions(block: DocumentBlock, regions: list[object]) -> bool:
    if block.bbox is None:
        return False
    import fitz  # type: ignore

    bbox = fitz.Rect(*block.bbox.to_list())
    text = _block_text(block)
    for region in regions:
        center_x = (bbox.x0 + bbox.x1) / 2.0
        center_y = (bbox.y0 + bbox.y1) / 2.0
        center_inside = region.x0 - 6.0 <= center_x <= region.x1 + 6.0 and region.y0 - 6.0 <= center_y <= region.y1 + 6.0
        overlap = _algorithm_overlap_ratio(bbox, region)
        if center_inside and overlap > 0.12:
            return True
        if _looks_like_algorithm_text(text) and overlap > 0.08:
            return True
    return False


def _algorithm_overlap_ratio(left: object, right: object) -> float:
    inter = left & right
    if inter.is_empty or left.get_area() <= 0:
        return 0.0
    return float(inter.get_area() / left.get_area())


def _block_text(block: DocumentBlock) -> str:
    return re.sub(r"\s+", " ", (block.text or block.latex or block.caption or "")).strip()


def _position_key(block: DocumentBlock) -> tuple[int, float, float]:
    if block.bbox is None:
        return (int(block.page or 0), float("inf"), float("inf"))
    return (int(block.page or 0), block.bbox.y0, block.bbox.x0)
