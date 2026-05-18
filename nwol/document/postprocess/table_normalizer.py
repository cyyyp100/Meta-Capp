from __future__ import annotations

import hashlib
import html
import logging
import re
from pathlib import Path

from document.models import BoundingBox, DocumentBlock
from document.postprocess.figure_extractor import document_asset_dir

logger = logging.getLogger("Document.table_normalizer")


def extract_native_tables(
    pdf_path: str,
    blocks: list[DocumentBlock],
    *,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    """
    Tente d'extraire les tableaux via PyMuPDF find_tables().
    Retourne les blocks avec les blocs tableau remplacés par des blocs haute confiance.
    Si find_tables() est indisponible ou ne trouve rien, retourne blocks inchangé.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        return blocks

    if not hasattr(fitz.Page, "find_tables"):
        return blocks

    path = Path(pdf_path)
    replaced_ids: set[int] = set()
    native_tables: list[DocumentBlock] = []

    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                if pages is not None and page_index not in pages:
                    continue
                page_blocks = [block for block in blocks if int(block.page or 0) == page_index]
                try:
                    finder = page.find_tables()
                except Exception:
                    continue

                for table in finder.tables:
                    row_count = int(getattr(table, "row_count", 0) or 0)
                    col_count = int(getattr(table, "col_count", 0) or 0)
                    if (row_count and row_count < 2) or (col_count and col_count < 2):
                        continue

                    rows = table.extract()
                    if not rows:
                        continue

                    cleaned = [
                        [_normalize_cell(str(cell or "")) for cell in (row or [])]
                        for row in rows
                        if row is not None
                    ]
                    cleaned = [row for row in cleaned if any(row)]
                    if not cleaned:
                        continue

                    max_cols = max(len(row) for row in cleaned)
                    if len(cleaned) < 2 or max_cols < 2:
                        continue

                    cleaned = [row + [""] * (max_cols - len(row)) for row in cleaned]
                    markdown = _to_markdown(cleaned)
                    text = "\n".join(" | ".join(row) for row in cleaned)
                    rect = table.bbox
                    if rect is None:
                        continue
                    bbox = BoundingBox(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))

                    for block in page_blocks:
                        if block.bbox and _bbox_overlap_ratio(block.bbox, bbox) > 0.5:
                            replaced_ids.add(id(block))

                    table_image_path = _crop_table_image(page, rect, pdf_path, page_index)

                    native_tables.append(
                        DocumentBlock(
                            type="table",
                            text=text,
                            page=page_index,
                            bbox=bbox,
                            markdown=markdown,
                            html=_to_html(cleaned),
                            confidence=0.92,
                            metadata={
                                "rows": len(cleaned),
                                "columns": max_cols,
                                "source": "pymupdf_native",
                                "table_image_path": table_image_path,
                            },
                        )
                    )
    except Exception as exc:
        logger.warning("extract_native_tables échoué: %s", exc)
        return blocks

    if not native_tables:
        return blocks

    kept = [block for block in blocks if id(block) not in replaced_ids]
    kept.extend(native_tables)
    return sorted(kept, key=lambda block: (int(block.page or 0), block.bbox.y0 if block.bbox else 0, block.bbox.x0 if block.bbox else 0))


def crop_rule_based_tables(
    pdf_path: str,
    blocks: list[DocumentBlock],
    *,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    """Recover ruled tables missed by PyMuPDF's native table detector.

    Some LaTeX tables expose only horizontal rules plus normal text blocks.
    Without a reconstructed table block, individual cells can leak into the
    reader flow after the following heading.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        return blocks

    path = Path(pdf_path)
    removed_ids: set[int] = set()
    recovered_tables: list[DocumentBlock] = []

    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                if pages is not None and page_index not in pages:
                    continue
                page_blocks = [block for block in blocks if int(block.page or 0) == page_index]
                for table_index, rect in enumerate(_rule_based_table_rects(page)):
                    bbox = BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                    if _overlaps_existing_table(page_blocks, bbox):
                        continue

                    caption = _nearest_table_caption(page_blocks, bbox)
                    table_text = _extract_table_text(page, rect)
                    table_image_path = _crop_table_image(page, bbox.to_list(), pdf_path, page_index)
                    if not table_image_path and not table_text:
                        continue

                    remove_rect = bbox
                    if caption is not None and caption.bbox is not None:
                        remove_rect = remove_rect.union(caption.bbox)
                    for block in page_blocks:
                        if _block_belongs_to_rule_based_table(block, bbox, caption):
                            removed_ids.add(id(block))

                    recovered_tables.append(
                        DocumentBlock(
                            type="table",
                            text=table_text,
                            page=page_index,
                            bbox=remove_rect,
                            image_path=None,
                            caption=(caption.text.strip() if caption and caption.text else None),
                            confidence=0.82,
                            metadata={
                                "source": "rule_based_horizontal_lines",
                                "table_index": table_index,
                                "table_image_path": table_image_path,
                                "caption_display": True,
                            },
                        )
                    )
    except Exception as exc:
        logger.warning("crop_rule_based_tables échoué: %s", exc)
        return blocks

    if not recovered_tables:
        return blocks
    kept = [block for block in blocks if id(block) not in removed_ids]
    kept.extend(recovered_tables)
    return sorted(kept, key=lambda block: (int(block.page or 0), block.bbox.y0 if block.bbox else 0, block.bbox.x0 if block.bbox else 0))


def normalize_tables(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    table_run: list[DocumentBlock] = []

    def flush() -> None:
        nonlocal table_run
        if len(table_run) >= 2:
            result.append(_build_table(table_run))
        else:
            result.extend(table_run)
        table_run = []

    for block in blocks:
        if block.type == "paragraph" and _looks_like_table_line(block.text):
            table_run.append(block)
            continue
        flush()
        result.append(block)
    flush()
    return result


def crop_table_blocks(
    pdf_path: str,
    blocks: list[DocumentBlock],
    *,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    """Attach a PDF crop image to every table block with usable geometry."""
    candidates = [
        block
        for block in blocks
        if block.type == "table"
        and block.bbox is not None
        and block.page is not None
        and (pages is None or int(block.page or 0) in pages)
    ]
    if not candidates:
        return blocks

    try:
        import fitz  # type: ignore
    except Exception:
        return blocks

    try:
        with fitz.open(pdf_path) as doc:
            for block in candidates:
                metadata = block.metadata if isinstance(block.metadata, dict) else {}
                block.metadata = metadata

                existing = metadata.get("table_image_path")
                if existing:
                    try:
                        if Path(str(existing)).expanduser().exists():
                            continue
                    except OSError:
                        pass

                page_index = int(block.page or 1)
                if page_index < 1 or page_index > len(doc):
                    continue

                image_path = _crop_table_image(
                    doc[page_index - 1],
                    block.bbox.to_list(),
                    pdf_path,
                    page_index,
                )
                if image_path:
                    metadata["table_image_path"] = image_path
                    metadata.setdefault("table_image_source", "pdf_bbox_crop")
    except Exception as exc:
        logger.debug("Crop des tableaux ignoré: %s", exc)
        return blocks

    return blocks


def _normalize_cell(cell: str) -> str:
    cell = cell.replace("\n", " ").replace("\r", " ")
    cell = re.sub(r" {2,}", " ", cell)
    return cell.strip()


def _looks_like_table_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) < 10:
        return False
    if stripped.count("|") >= 2:
        parts = [p.strip() for p in stripped.split("|") if p.strip()]
        return len(parts) >= 2
    if "\t" in stripped:
        parts = [p.strip() for p in stripped.split("\t") if p.strip()]
        return len(parts) >= 2
    columns = re.split(r"\s{2,}", stripped)
    return len([col for col in columns if col.strip()]) >= 3


def _split_row(text: str) -> list[str]:
    stripped = text.strip().strip("|")
    if "|" in stripped:
        return [_normalize_cell(cell) for cell in stripped.split("|")]
    if "\t" in stripped:
        return [_normalize_cell(cell) for cell in stripped.split("\t")]
    return [_normalize_cell(cell) for cell in re.split(r"\s{2,}", stripped) if cell.strip()]


def _build_table(blocks: list[DocumentBlock]) -> DocumentBlock:
    rows = [_split_row(block.text) for block in blocks]
    max_cols = max((len(row) for row in rows), default=0)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]
    text = "\n".join(" | ".join(row).strip() for row in rows)
    markdown = _to_markdown(rows)
    html_table = _to_html(rows)
    bbox: BoundingBox | None = blocks[0].bbox
    for block in blocks[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
    return DocumentBlock(
        type="table",
        text=text,
        page=blocks[0].page,
        bbox=bbox,
        markdown=markdown,
        html=html_table,
        confidence=min(block.confidence for block in blocks),
        metadata={"rows": len(rows), "columns": max_cols, "source": "table_normalizer"},
    )


def _to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])


def _to_html(rows: list[list[str]]) -> str:
    if not rows:
        return "<table></table>"
    html_rows = []
    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        cells = "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in row)
        html_rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def _rule_based_table_rects(page: object) -> list[object]:
    rules = _horizontal_rule_rects(page)
    if len(rules) < 3:
        return []

    import fitz  # type: ignore

    groups: list[list[object]] = []
    for rule in sorted(rules, key=lambda rect: (rect.y0, rect.x0)):
        matched = None
        for group in groups:
            if _rules_compatible(group[0], rule):
                matched = group
                break
        if matched is None:
            groups.append([rule])
        else:
            matched.append(rule)

    rects: list[object] = []
    for group in groups:
        group = sorted(group, key=lambda rect: rect.y0)
        if len(group) < 3:
            continue
        y_span = group[-1].y0 - group[0].y0
        if y_span < 24.0 or y_span > 260.0:
            continue
        rect = fitz.Rect(
            min(rule.x0 for rule in group),
            min(rule.y0 for rule in group),
            max(rule.x1 for rule in group),
            max(rule.y1 for rule in group),
        )
        if rect.width >= 180.0:
            rects.append(rect)
    return rects


def _horizontal_rule_rects(page: object) -> list[object]:
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    rules = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        drawing_type = str(drawing.get("type") or "")
        if "s" not in drawing_type:
            continue
        if rect.width >= 160.0 and rect.height <= 2.0:
            rules.append(rect)
    return rules


def _rules_compatible(left: object, right: object) -> bool:
    overlap = min(left.x1, right.x1) - max(left.x0, right.x0)
    if overlap <= min(left.width, right.width) * 0.72:
        return False
    return abs(left.x0 - right.x0) <= 24.0 and abs(left.x1 - right.x1) <= 24.0


def _overlaps_existing_table(blocks: list[DocumentBlock], bbox: BoundingBox) -> bool:
    for block in blocks:
        if block.type == "table" and block.bbox is not None and _bbox_overlap_ratio(block.bbox, bbox) > 0.25:
            return True
    return False


def _nearest_table_caption(blocks: list[DocumentBlock], table_bbox: BoundingBox) -> DocumentBlock | None:
    candidates: list[tuple[float, DocumentBlock]] = []
    for block in blocks:
        if block.bbox is None:
            continue
        text = (block.text or block.caption or "").strip()
        metadata = block.metadata or {}
        if not (metadata.get("is_caption") or re.match(r"^(?:table|tableau)\s+\d", text, re.I)):
            continue
        if block.bbox.y0 < table_bbox.y1:
            continue
        distance = block.bbox.y0 - table_bbox.y1
        if distance > 55.0:
            continue
        horizontal_overlap = min(block.bbox.x1, table_bbox.x1) - max(block.bbox.x0, table_bbox.x0)
        if horizontal_overlap <= min(block.bbox.width, table_bbox.width) * 0.12:
            continue
        candidates.append((distance, block))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _extract_table_text(page: object, rect: object) -> str:
    try:
        blocks = page.get_text("blocks", clip=rect)
    except Exception:
        return ""
    parts = []
    for block in sorted(blocks, key=lambda item: (float(item[1]), float(item[0]))):
        text = str(block[4] or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _block_belongs_to_rule_based_table(
    block: DocumentBlock,
    table_bbox: BoundingBox,
    caption: DocumentBlock | None = None,
) -> bool:
    if block.bbox is None:
        return False
    if caption is not None and id(block) == id(caption):
        return True
    if block.type in {"figure", "table"}:
        return False
    bbox = block.bbox
    expanded = BoundingBox(table_bbox.x0 - 8.0, table_bbox.y0 - 8.0, table_bbox.x1 + 8.0, table_bbox.y1 + 8.0)
    center_inside = expanded.x0 <= bbox.center_x <= expanded.x1 and expanded.y0 <= bbox.center_y <= expanded.y1
    if block.type == "formula":
        return center_inside or _bbox_overlap_ratio(bbox, expanded) > 0.45
    if center_inside:
        return True
    if _bbox_overlap_ratio(bbox, expanded) > 0.35:
        return True
    metadata = block.metadata or {}
    if metadata.get("semantic_only_block") and _bbox_overlap_ratio(expanded, bbox) > 0.15:
        return True
    return False


def _bbox_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = a.width * a.height
    return inter / area_a if area_a > 0 else 0.0


def _crop_table_image(page, rect, pdf_path: str, page_index: int) -> str | None:
    """Crop le rectangle du tableau depuis la page PDF et sauvegarde en PNG. Retourne le chemin ou None."""
    try:
        import fitz  # type: ignore

        output_dir = document_asset_dir(pdf_path) / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)

        fitz_rect = fitz.Rect(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
        page_rect = page.rect
        fitz_rect = fitz.Rect(
            max(fitz_rect.x0 - 4, page_rect.x0),
            max(fitz_rect.y0 - 4, page_rect.y0),
            min(fitz_rect.x1 + 4, page_rect.x1),
            min(fitz_rect.y1 + 4, page_rect.y1),
        )
        if fitz_rect.is_empty or fitz_rect.width <= 10 or fitz_rect.height <= 10:
            return None

        h = hashlib.sha1(f"{pdf_path}_{page_index}_{rect}".encode()).hexdigest()[:8]
        img_path = output_dir / f"table_p{page_index}_{h}.png"
        if not img_path.exists():
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), clip=fitz_rect, alpha=False)
            pix.save(str(img_path))
        return str(img_path)
    except Exception as exc:
        logger.debug("Crop tableau ignoré p.%s: %s", page_index, exc)
        return None
