from __future__ import annotations

import json
import logging
from pathlib import Path

from document.models import DocumentBlock, ExtractionResult

logger = logging.getLogger("Document.debug")

COLORS = {
    "heading": (128, 0, 180),
    "paragraph": (30, 100, 220),
    "formula": (220, 30, 30),
    "figure": (30, 150, 70),
    "table": (230, 140, 20),
    "bullet_list": (0, 170, 180),
}


def export_layout_debug(
    result: ExtractionResult,
    pdf_path: str,
    output_dir: str | Path,
) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    blocks_path = output / "document_blocks.json"
    blocks_path.write_text(
        json.dumps(result.to_reader_blocks(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.append(str(blocks_path))

    report_path = output / "extraction_report.json"
    report_path.write_text(
        json.dumps(
            {
                "engine": result.engine_name,
                "score": result.score,
                "warnings": result.warnings,
                "pages": result.pages,
                "blocks": len(result.blocks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.append(str(report_path))

    try:
        paths.extend(_export_page_images(result.blocks, pdf_path, output))
    except Exception as exc:
        logger.debug("Export visuel indisponible: %s", exc)

    result.debug_paths = paths
    return paths


def _export_page_images(blocks: list[DocumentBlock], pdf_path: str, output: Path) -> list[str]:
    import fitz  # type: ignore
    from PIL import Image, ImageDraw, ImageFont

    paths: list[str] = []
    blocks_by_page: dict[int, list[tuple[int, DocumentBlock]]] = {}
    for index, block in enumerate(blocks, start=1):
        if block.page is None or block.bbox is None:
            continue
        blocks_by_page.setdefault(int(block.page), []).append((index, block))

    with fitz.open(pdf_path) as doc:
        for page_number, page_blocks in blocks_by_page.items():
            if page_number < 1 or page_number > len(doc):
                continue
            page = doc[page_number - 1]
            scale = 144 / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            for order, block in page_blocks:
                bbox = block.bbox
                if bbox is None:
                    continue
                color = COLORS.get(block.type, (90, 90, 90))
                rect = [bbox.x0 * scale, bbox.y0 * scale, bbox.x1 * scale, bbox.y1 * scale]
                draw.rectangle(rect, outline=color, width=2)
                draw.text((rect[0] + 2, rect[1] + 2), str(order), fill=color, font=font)
            page_path = output / f"page_{page_number}_layout.png"
            image.save(page_path)
            paths.append(str(page_path))
    return paths
