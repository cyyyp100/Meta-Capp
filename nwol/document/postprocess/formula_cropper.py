from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from document.models import DocumentBlock
from document.postprocess.figure_extractor import document_asset_dir

logger = logging.getLogger("Document.formula_cropper")


def crop_formula_blocks(
    pdf_path: str,
    blocks: list[DocumentBlock],
    output_dir: str | Path | None = None,
) -> list[DocumentBlock]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        logger.debug("PyMuPDF indisponible pour crop formules: %s", exc)
        return blocks

    path = Path(pdf_path)
    out = Path(output_dir) if output_dir is not None else document_asset_dir(path) / "formulas"
    out.mkdir(parents=True, exist_ok=True)

    try:
        with fitz.open(path) as doc:
            for index, block in enumerate(blocks):
                if block.type != "formula":
                    continue
                if not block.bbox or not block.page:
                    continue
                if not _should_crop_formula(block):
                    continue

                page_index = block.page - 1
                if page_index < 0 or page_index >= len(doc):
                    continue
                page = doc[page_index]
                rect = fitz.Rect(*block.bbox.to_list())
                rect = _expand_formula_rect(rect, block, page.rect)
                rect, bottom_limited = _trim_formula_bottom_against_text(rect, block, blocks)
                pad_x = 14.0
                pad_top = 6.0
                pad_bottom = 0.0 if bottom_limited else 6.0
                rect = fitz.Rect(rect.x0 - pad_x, rect.y0 - pad_top, rect.x1 + pad_x, rect.y1 + pad_bottom)
                rect = fitz.Rect(
                    max(rect.x0, page.rect.x0),
                    max(rect.y0, page.rect.y0),
                    min(rect.x1, page.rect.x1),
                    min(rect.y1, page.rect.y1),
                )
                if rect.is_empty or rect.width <= 1 or rect.height <= 1:
                    continue

                _CROP_VERSION = "v5_trimmed"
                digest = hashlib.md5(f"{path}-{block.page}-{block.bbox.to_list()}-{_CROP_VERSION}".encode()).hexdigest()[:12]
                image_path = out / f"formula_p{block.page}_{index}_{digest}.png"
                if not image_path.exists():
                    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(4, 4), alpha=False)
                    if not _save_trimmed_formula_pixmap(pix, image_path):
                        pix.save(str(image_path))

                block.image_path = str(image_path)
                block.metadata["formula_image_path"] = str(image_path)
                block.metadata["render_mode"] = "pdf_crop"
    except Exception as exc:
        logger.warning("Crop des formules PDF échoué: %s", exc)

    return blocks


def _save_trimmed_formula_pixmap(pix, image_path: Path) -> bool:
    try:
        from PIL import Image

        mode = "RGBA" if getattr(pix, "alpha", False) else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        image = _trim_light_background(image, padding=12)
        image.save(image_path)
        return True
    except Exception as exc:
        logger.debug("Rognage formule ignoré pour %s: %s", image_path, exc)
        return False


def _trim_light_background(image, *, padding: int = 8):
    try:
        gray = image.convert("L")
        mask = gray.point(lambda pixel: 255 if pixel < 248 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return image

        left, top, right, bottom = bbox
        if right - left < 2 or bottom - top < 2:
            return image
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)
        if left == 0 and top == 0 and right == image.width and bottom == image.height:
            return image
        return image.crop((left, top, right, bottom))
    except Exception:
        return image


def _expand_formula_rect(rect, block: DocumentBlock, page_rect):
    metadata = block.metadata or {}
    if not metadata.get("wide_initial_crop"):
        return rect

    page_width = float(page_rect.width)
    if page_width <= 0:
        return rect

    target_width = min(page_width * 0.72, max(rect.width + 180.0, page_width * 0.36))
    extra = max(0.0, target_width - rect.width)
    return type(rect)(
        max(float(page_rect.x0), rect.x0 - extra * 0.45),
        max(float(page_rect.y0), rect.y0 - 2.0),
        min(float(page_rect.x1), rect.x1 + extra * 0.55),
        min(float(page_rect.y1), rect.y1 + 2.0),
    )


def _trim_formula_bottom_against_text(rect, block: DocumentBlock, blocks: list[DocumentBlock]):
    """Avoid formula crops swallowing the prose line immediately below them."""
    if block.page is None:
        return rect, False
    nearest_y0: float | None = None
    for other in blocks:
        if other is block or other.page != block.page or other.bbox is None:
            continue
        if other.type in {"formula", "figure", "table"}:
            continue
        metadata = other.metadata or {}
        if metadata.get("formula_mode") == "display":
            continue
        other_rect = other.bbox
        if other_rect.x1 <= rect.x0 or other_rect.x0 >= rect.x1:
            continue
        if other_rect.y0 <= rect.y0:
            continue
        if other_rect.y0 > rect.y1 + 8.0:
            continue
        nearest_y0 = other_rect.y0 if nearest_y0 is None else min(nearest_y0, other_rect.y0)

    if nearest_y0 is None:
        return rect, False
    trimmed_y1 = max(rect.y0 + 4.0, min(rect.y1, nearest_y0 - 1.0))
    return type(rect)(rect.x0, rect.y0, rect.x1, trimmed_y1), True


_CITATION_ONLY_RE = re.compile(r"^\$?\s*\[[\d,\s]+\]\.?\s*\$?$")


def _should_crop_formula(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    if metadata.get("formula_mode") == "inline":
        return False
    text = (block.text or block.latex or "").strip()
    if _CITATION_ONLY_RE.match(text):
        return False
    if metadata.get("render_mode") == "pdf_crop" and metadata.get("formula_mode") == "display":
        return True
    if metadata.get("render_mode") == "pdf_crop" and not block.image_path:
        return True
    if metadata.get("formula_mode") == "display" and _has_reasonable_formula_bbox(block):
        return True
    if block.latex and _has_reasonable_formula_bbox(block) and metadata.get("formula_mode") != "inline":
        return True

    if not text:
        return False
    if "$" in text and text.count("$") != 2:
        return True
    if re.search(r"\$[A-Za-zÀ-ÿ]{1,2}\$|\bl\s*\$\s*n\b", text):
        return True
    if text.count("(") - text.count(")") >= 2:
        return True
    return False


def _has_reasonable_formula_bbox(block: DocumentBlock) -> bool:
    if block.bbox is None:
        return False
    if block.bbox.width <= 4 or block.bbox.height <= 4:
        return False
    if block.bbox.width > 850 or block.bbox.height > 300:
        return False
    return True
