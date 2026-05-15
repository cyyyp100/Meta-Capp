from __future__ import annotations

import json
import logging
import re
from html import escape
from pathlib import Path
from typing import Any

from document.extractors.base import BaseExtractor, OptionalBackendUnavailable, markdown_to_document_blocks
from document.layout.block_classifier import deduplicate_heading_blocks, promote_short_keyword_headings
from document.models import BoundingBox, DocumentBlock, ExtractionResult
from document.postprocess.figure_extractor import document_asset_dir
from document.postprocess.pipeline import postprocess_document_blocks
from document.postprocess.quality import update_result_quality

logger = logging.getLogger("Document.OpenDataLoader")


class OpenDataLoaderExtractor(BaseExtractor):
    engine_name = "opendataloader_pdf"

    def extract(self, pdf_path: str) -> ExtractionResult:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF introuvable: {path}")

        page_sizes = _detect_page_sizes(path)
        data, markdown, debug_paths, output_dir = run_opendataloader(path)

        blocks = convert_opendataloader_json_to_document_blocks(
            data,
            markdown=markdown,
            base_path=output_dir,
            page_sizes=page_sizes,
        )

        logger.info(
            "OpenDataLoader pages couvertes avant postprocess: %s",
            sorted({block.page for block in blocks if block.page}),
        )

        if not blocks and markdown:
            blocks = markdown_to_document_blocks(markdown, base_path=output_dir)
            for block in blocks:
                block.metadata.setdefault("source", self.engine_name)

        pages = detect_page_count(data, blocks, page_sizes)

        warnings: list[str] = []
        if not blocks:
            warnings.append("Aucun bloc lisible extrait par OpenDataLoader.")

        blocks = postprocess_document_blocks(blocks, str(path), page_sizes)

        logger.info(
            "OpenDataLoader pages couvertes après postprocess: %s",
            sorted({block.page for block in blocks if block.page}),
        )

        for index, block in enumerate(blocks):
            block.metadata.setdefault("source", self.engine_name)
            block.metadata.setdefault("engine", self.engine_name)
            block.metadata.setdefault("block_index", index)
            if block.page in page_sizes:
                width, height = page_sizes[block.page]
                block.metadata.setdefault("page_width", width)
                block.metadata.setdefault("page_height", height)

        result = ExtractionResult(
            blocks=blocks,
            pages=pages,
            score=0.0,
            warnings=warnings,
            engine_name=self.engine_name,
            debug_paths=debug_paths,
        )
        return update_result_quality(result)


def run_opendataloader(pdf_path: str | Path) -> tuple[Any, str | None, list[str], Path]:
    """Run opendataloader-pdf and return parsed JSON plus optional Markdown."""
    try:
        import opendataloader_pdf  # type: ignore
    except ImportError as exc:
        raise OptionalBackendUnavailable(
            "opendataloader-pdf non installé. Lance: pip install -U opendataloader-pdf"
        ) from exc

    convert = getattr(opendataloader_pdf, "convert", None)
    if convert is None:
        raise OptionalBackendUnavailable("opendataloader_pdf.convert est introuvable.")

    path = Path(pdf_path)
    output_dir = document_asset_dir(path) / "opendataloader"
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "input_path": str(path),
        "output_dir": str(output_dir),
        "format": "json,markdown",
        "quiet": True,
        "use_struct_tree": True,
        "table_method": "cluster",
        "reading_order": "xycut",
    }

    try:
        converted = _call_convert(convert, kwargs)
    except OptionalBackendUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError(f"OpenDataLoader a échoué: {exc}") from exc

    data, markdown, debug_paths = _collect_outputs(converted, output_dir, path)
    if data is None:
        raise RuntimeError("OpenDataLoader n'a produit aucun JSON exploitable.")

    return data, markdown, debug_paths, output_dir


def convert_opendataloader_json_to_document_blocks(
    data: Any,
    *,
    markdown: str | None = None,
    base_path: str | Path | None = None,
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []

    for node in _top_level_nodes(data):
        blocks.extend(
            _convert_node(
                node,
                base_path=Path(base_path) if base_path else None,
                page_sizes=page_sizes,
            )
        )

    if not blocks and markdown:
        blocks = markdown_to_document_blocks(markdown, base_path=base_path)

    blocks = promote_short_keyword_headings(blocks)
    blocks = deduplicate_heading_blocks(blocks)

    for index, block in enumerate(blocks):
        block.metadata.setdefault("source", OpenDataLoaderExtractor.engine_name)
        block.metadata.setdefault("opendataloader_index", index)

    return blocks


def detect_page_count(
    data: Any,
    blocks: list[DocumentBlock],
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> int:
    if isinstance(data, dict):
        for key in ("number of pages", "number_of_pages", "pages", "page_count", "pageCount"):
            value = data.get(key)

            if isinstance(value, int):
                return max(0, value)

            if isinstance(value, list):
                return len(value)

            try:
                if value is not None:
                    return max(0, int(value))
            except (TypeError, ValueError):
                pass

    if page_sizes:
        return len(page_sizes)

    return max((int(block.page or 0) for block in blocks), default=0)


def _call_convert(convert: Any, kwargs: dict[str, Any]) -> Any:
    optional_keys = ("reading_order", "table_method", "use_struct_tree", "quiet")
    last_exc: Exception | None = None
    current = dict(kwargs)

    for key_to_remove in (None, *optional_keys):
        if key_to_remove is not None:
            current.pop(key_to_remove, None)

        try:
            return convert(**current)
        except TypeError as exc:
            message = str(exc)
            last_exc = exc

            if not _looks_like_unsupported_kwarg(message):
                raise

            continue

    if last_exc is not None:
        raise last_exc

    return convert(**kwargs)


def _looks_like_unsupported_kwarg(message: str) -> bool:
    lowered = message.lower()
    return (
        "unexpected keyword" in lowered
        or "got an unexpected" in lowered
        or "invalid keyword" in lowered
    )


def _collect_outputs(converted: Any, output_dir: Path, pdf_path: Path) -> tuple[Any | None, str | None, list[str]]:
    data = _data_from_return_value(converted)
    debug_paths: list[str] = []

    json_path = _find_output_file(output_dir, pdf_path.stem, {".json"})
    if data is None and json_path is not None:
        data = json.loads(json_path.read_text(encoding="utf-8"))

    if json_path is not None:
        debug_paths.append(str(json_path))

    markdown: str | None = None
    md_path = _find_output_file(output_dir, pdf_path.stem, {".md", ".markdown"})
    if md_path is not None:
        markdown = md_path.read_text(encoding="utf-8")
        debug_paths.append(str(md_path))

    return data, markdown, debug_paths


def _data_from_return_value(value: Any) -> Any | None:
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return value

        for item in value:
            data = _data_from_return_value(item)
            if data is not None:
                return data

    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == ".json" and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    return None


def _find_output_file(output_dir: Path, stem: str, suffixes: set[str]) -> Path | None:
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]

    if not candidates:
        return None

    normalized_stem = _normalize_name(stem)

    exact = [
        path
        for path in candidates
        if _normalize_name(path.stem) == normalized_stem
    ]
    if exact:
        return max(exact, key=lambda path: path.stat().st_mtime)

    containing = [
        path
        for path in candidates
        if normalized_stem in _normalize_name(path.stem)
    ]
    if containing:
        return max(containing, key=lambda path: path.stat().st_mtime)

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _top_level_nodes(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [node for node in data if isinstance(node, dict)]

    if not isinstance(data, dict):
        return []

    for key in ("kids", "children", "elements", "blocks", "content"):
        value = data.get(key)
        if isinstance(value, list):
            return [node for node in value if isinstance(node, dict)]

    pages = data.get("pages")
    if isinstance(pages, list):
        result: list[dict[str, Any]] = []
        for index, page_node in enumerate(pages, start=1):
            if isinstance(page_node, dict):
                page_node = dict(page_node)
                page_node.setdefault("page number", index)
                result.append(page_node)
        return result

    return [data]


def _convert_node(
    node: dict[str, Any],
    *,
    base_path: Path | None,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> list[DocumentBlock]:
    node_type = _node_type(node)

    if node_type in {"document", "page", "text block", "textbox", "section", "article"}:
        return _convert_children(node, base_path=base_path, page_sizes=page_sizes)

    if node_type in {"header", "footer"}:
        return []

    if node_type == "table":
        return [_convert_table(node, page_sizes=page_sizes)]

    if node_type in {"list", "ordered list", "unordered list"}:
        return _convert_list(
            node,
            base_path=base_path,
            page_sizes=page_sizes,
        )

    if node_type in {"image", "figure", "picture"}:
        return [_convert_figure(node, base_path=base_path, page_sizes=page_sizes)]

    if node_type in {"formula", "equation", "math"}:
        return [_convert_formula(node, page_sizes=page_sizes)]

    if node_type in {"heading", "title"}:
        return [_convert_text_node(node, "heading", page_sizes=page_sizes)]

    if node_type == "caption":
        block = _convert_text_node(node, "paragraph", page_sizes=page_sizes)
        block.metadata["is_caption"] = True
        return [block]

    if node_type in {"paragraph", "text", "list item"}:
        return [_convert_text_node(node, "paragraph", page_sizes=page_sizes)]

    children = _convert_children(node, base_path=base_path, page_sizes=page_sizes)
    if children:
        return children

    text = _node_text(node)
    if not text:
        return []

    return [_convert_text_node(node, "paragraph", page_sizes=page_sizes)]


def _convert_children(
    node: dict[str, Any],
    *,
    base_path: Path | None,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    parent_page = _page_number(node)

    for key in ("kids", "children", "elements", "blocks", "content", "list items"):
        value = node.get(key)
        if not isinstance(value, list):
            continue

        for child in value:
            if isinstance(child, dict):
                child = dict(child)

                if parent_page is not None and _page_number(child) is None:
                    child["page number"] = parent_page

                result.extend(
                    _convert_node(
                        child,
                        base_path=base_path,
                        page_sizes=page_sizes,
                    )
                )

        if result:
            return result

    return result


def _convert_text_node(
    node: dict[str, Any],
    block_type: str,
    *,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> DocumentBlock:
    text = _node_text(node)
    metadata = _node_metadata(node)
    level = _heading_level(node) if block_type == "heading" else None

    return DocumentBlock(
        type=block_type,
        text=text,
        page=_page_number(node),
        bbox=_bbox(node, page_sizes=page_sizes),
        level=level,
        confidence=1.0,
        metadata=metadata,
        id=_node_id(node),
    )


def _convert_formula(
    node: dict[str, Any],
    *,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> DocumentBlock:
    latex = _first_text(node, ("latex", "LaTeX", "tex", "formula", "content", "text"))
    text = latex or _node_text(node)

    return DocumentBlock(
        type="formula",
        text=text,
        page=_page_number(node),
        bbox=_bbox(node, page_sizes=page_sizes),
        latex=latex or text,
        confidence=1.0,
        metadata={**_node_metadata(node), "formula_mode": "display"},
        id=_node_id(node),
    )


def _convert_figure(
    node: dict[str, Any],
    *,
    base_path: Path | None,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> DocumentBlock:
    image_path = _image_path(node, base_path=base_path)
    caption = _first_text(node, ("caption", "alt", "content", "text"))

    return DocumentBlock(
        type="figure",
        text=caption or "",
        page=_page_number(node),
        bbox=_bbox(node, page_sizes=page_sizes),
        image_path=image_path,
        caption=caption or "",
        confidence=0.9,
        metadata=_node_metadata(node),
        id=_node_id(node),
    )


_SECTION_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)+|[A-Z](?:\.\d+)+|[A-Z]\.)\.?\s+[A-Za-zÀ-ÿ]",
    re.I,
)


def _convert_list(
    node: dict[str, Any],
    *,
    base_path: Path | None,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> list[DocumentBlock]:
    item_nodes = _list_item_nodes(node)

    if _looks_like_structural_section_list(item_nodes):
        return _convert_structural_section_list(
            item_nodes,
            base_path=base_path,
            page_sizes=page_sizes,
            parent_page=_page_number(node),
        )

    return [_convert_bullet_list(node, item_nodes=item_nodes, page_sizes=page_sizes)]


def _convert_bullet_list(
    node: dict[str, Any],
    *,
    item_nodes: list[Any],
    page_sizes: dict[int, tuple[float, float]] | None,
) -> DocumentBlock:

    items: list[str] = []

    for item in item_nodes:
        if isinstance(item, dict):
            text = _node_text(item)
        else:
            text = str(item)

        text = re.sub(r"\s+", " ", text).strip()
        if text:
            items.append(text)

    if not items:
        text = _node_text(node)
        items = [text] if text else []

    return DocumentBlock(
        type="bullet_list",
        text="\n".join(f"• {item}" for item in items),
        page=_page_number(node),
        bbox=_bbox(node, page_sizes=page_sizes),
        items=items,
        confidence=1.0,
        metadata=_node_metadata(node),
        id=_node_id(node),
    )


def _list_item_nodes(node: dict[str, Any]) -> list[Any]:
    value = (
        node.get("list items")
        or node.get("list_items")
        or node.get("items")
        or node.get("kids")
        or []
    )
    return list(value) if isinstance(value, list) else []


def _looks_like_structural_section_list(item_nodes: list[Any]) -> bool:
    for item in item_nodes:
        if isinstance(item, dict):
            if _looks_like_section_list_item(_direct_node_text(item)):
                return True
            for child in _direct_child_nodes(item):
                if isinstance(child, dict) and _looks_like_section_list_item(_direct_node_text(child)):
                    return True
        elif _looks_like_section_list_item(str(item)):
            return True
    return False


def _convert_structural_section_list(
    item_nodes: list[Any],
    *,
    base_path: Path | None,
    page_sizes: dict[int, tuple[float, float]] | None,
    parent_page: int | None,
) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for item in item_nodes:
        if not isinstance(item, dict):
            text = re.sub(r"\s+", " ", str(item)).strip()
            if text:
                result.append(DocumentBlock(type="paragraph", text=text, page=parent_page, confidence=1.0))
            continue

        node = dict(item)
        if parent_page is not None and _page_number(node) is None:
            node["page number"] = parent_page

        direct_text = _direct_node_text(node)
        if direct_text:
            block = _convert_text_node(node, "paragraph", page_sizes=page_sizes)
            block.text = direct_text
            block.metadata.setdefault("structural_list_item", True)
            result.append(block)

        for child in _direct_child_nodes(node):
            if not isinstance(child, dict):
                continue
            child = dict(child)
            if _page_number(node) is not None and _page_number(child) is None:
                child["page number"] = _page_number(node)
            result.extend(_convert_node(child, base_path=base_path, page_sizes=page_sizes))

    return result


def _looks_like_section_list_item(text: str) -> bool:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return False
    return bool(_SECTION_LIST_ITEM_RE.match(clean))


def _direct_node_text(node: dict[str, Any]) -> str:
    return _first_text(node, ("content", "text", "value", "markdown", "html", "caption", "alt", "label"))


def _direct_child_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in ("kids", "children", "elements", "blocks", "list items"):
        value = node.get(key)
        if isinstance(value, list):
            result.extend(child for child in value if isinstance(child, dict))
    return result


def _convert_table(
    node: dict[str, Any],
    *,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> DocumentBlock:
    rows = _table_rows(node)
    text = "\n".join(" | ".join(row) for row in rows)

    return DocumentBlock(
        type="table",
        text=text,
        page=_page_number(node),
        bbox=_bbox(node, page_sizes=page_sizes),
        markdown=_table_markdown(rows),
        html=_table_html(rows),
        confidence=1.0,
        metadata={
            **_node_metadata(node),
            "rows": len(rows),
            "columns": max((len(row) for row in rows), default=0),
        },
        id=_node_id(node),
    )


def _table_rows(node: dict[str, Any]) -> list[list[str]]:
    raw_rows = node.get("rows") or node.get("table rows") or node.get("table_rows") or []
    rows: list[list[str]] = []

    if isinstance(raw_rows, list):
        for row in raw_rows:
            if isinstance(row, dict):
                cells = row.get("cells") or row.get("kids") or []
            else:
                cells = row

            parsed = _table_cells(cells)
            if parsed:
                rows.append(parsed)

    if not rows:
        flat_cells = node.get("cells") or node.get("table cells") or node.get("table_cells")
        rows = _table_rows_from_flat_cells(flat_cells)

    if not rows:
        text = _node_text(node)
        for line in text.splitlines():
            cells = [
                cell.strip()
                for cell in re.split(r"\s{2,}|\|", line)
                if cell.strip()
            ]
            if cells:
                rows.append(cells)

    max_cols = max((len(row) for row in rows), default=0)
    return [row + [""] * (max_cols - len(row)) for row in rows]


def _table_cells(raw_cells: Any) -> list[str]:
    if not isinstance(raw_cells, list):
        return []

    cells: list[str] = []

    for cell in raw_cells:
        if isinstance(cell, dict):
            text = _node_text(cell)
        else:
            text = str(cell)

        cells.append(re.sub(r"\s+", " ", text).strip())

    return cells


def _table_rows_from_flat_cells(raw_cells: Any) -> list[list[str]]:
    if not isinstance(raw_cells, list):
        return []

    cells: list[tuple[int, int, str]] = []
    for index, cell in enumerate(raw_cells):
        if not isinstance(cell, dict):
            text = re.sub(r"\s+", " ", str(cell)).strip()
            if text:
                cells.append((0, index, text))
            continue

        row = _first_int(cell, ("row", "row_index", "row index", "r"), default=0)
        column = _first_int(cell, ("column", "col", "column_index", "column index", "c"), default=index)
        text = re.sub(r"\s+", " ", _node_text(cell)).strip()
        if text:
            cells.append((row, column, text))

    if not cells:
        return []

    row_count = max(row for row, _, _ in cells) + 1
    column_count = max(column for _, column, _ in cells) + 1
    rows = [["" for _ in range(column_count)] for _ in range(row_count)]
    for row, column, text in cells:
        rows[row][column] = text
    return [row for row in rows if any(cell.strip() for cell in row)]


def _table_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]

    return "\n".join([header, separator, *body])


def _table_html(rows: list[list[str]]) -> str:
    if not rows:
        return "<table></table>"

    html_rows: list[str] = []

    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        html_rows.append(
            "<tr>"
            + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row)
            + "</tr>"
        )

    return "<table>" + "".join(html_rows) + "</table>"


def _node_text(node: dict[str, Any]) -> str:
    direct = _first_text(node, ("content", "text", "value", "markdown", "html", "caption", "alt", "label"))
    if direct:
        return direct

    parts: list[str] = []

    for key in ("kids", "children", "elements", "blocks", "content", "spans", "lines"):
        value = node.get(key)
        if not isinstance(value, list):
            continue

        for child in value:
            text = _text_from_value(child)

            if text.strip():
                parts.append(text.strip())

    return " ".join(parts)


def _first_text(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _text_from_value(node.get(key))
        if text:
            return text

    return ""


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        parts = [_text_from_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()

    if isinstance(value, dict):
        for key in ("content", "text", "value", "markdown", "caption", "alt", "label"):
            text = _text_from_value(value.get(key))
            if text:
                return text

        parts: list[str] = []
        for key in ("kids", "children", "elements", "blocks", "spans", "lines", "content"):
            child_text = _text_from_value(value.get(key))
            if child_text:
                parts.append(child_text)
        return " ".join(parts).strip()

    return str(value).strip()


def _first_int(node: dict[str, Any], keys: tuple[str, ...], *, default: int = 0) -> int:
    for key in keys:
        value = node.get(key)
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return default


def _node_type(node: dict[str, Any]) -> str:
    return str(node.get("type") or node.get("role") or "").strip().casefold()


def _node_id(node: dict[str, Any]) -> str | None:
    for key in ("id", "content id", "content_id"):
        value = node.get(key)
        if value is not None:
            return str(value)

    return None


def _page_number(node: dict[str, Any]) -> int | None:
    candidates: list[Any] = []

    for key in (
        "page number",
        "page_number",
        "page",
        "pageNumber",
        "page_num",
        "pageNum",
        "page index",
        "page_index",
        "pageIndex",
    ):
        candidates.append(node.get(key))

    metadata = node.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "page number",
            "page_number",
            "page",
            "pageNumber",
            "page_num",
            "pageNum",
            "page index",
            "page_index",
            "pageIndex",
        ):
            candidates.append(metadata.get(key))

    for value in candidates:
        if value is None:
            continue

        if isinstance(value, dict):
            for nested_key in ("number", "page", "index"):
                nested = value.get(nested_key)
                try:
                    page = int(nested)
                    if nested_key == "index":
                        return page + 1 if page >= 0 else None
                    return max(1, page)
                except (TypeError, ValueError):
                    continue

        try:
            page = int(value)

            # Si OpenDataLoader renvoie des pages 0-indexées.
            if page == 0:
                return 1

            return max(1, page)
        except (TypeError, ValueError):
            continue

    return None


def _heading_level(node: dict[str, Any]) -> int | None:
    for key in ("heading level", "heading_level", "level"):
        value = node.get(key)

        try:
            level = int(value)
            return min(max(level, 1), 6)
        except (TypeError, ValueError):
            if isinstance(value, str) and value.strip().casefold() == "title":
                return 1

    return 1


def _bbox(
    node: dict[str, Any],
    *,
    page_sizes: dict[int, tuple[float, float]] | None,
) -> BoundingBox | None:
    for key in ("bounding box", "bounding_box", "boundingBox", "bbox"):
        value = node.get(key)
        parsed = _bbox_values(value, key)

        if parsed is not None:
            left, y_a, right, y_b, coordinate_system = parsed
            coordinate_system = _bbox_coordinate_system(node, key, coordinate_system)
            page = _page_number(node)

            if coordinate_system == "bottom_left" and page_sizes and page in page_sizes:
                page_height = page_sizes[page][1]
                y0 = page_height - max(y_a, y_b)
                y1 = page_height - min(y_a, y_b)
            else:
                y0 = min(y_a, y_b)
                y1 = max(y_a, y_b)

            return BoundingBox(
                min(left, right),
                y0,
                max(left, right),
                y1,
            )

    return None


def _bbox_values(value: Any, key: str) -> tuple[float, float, float, float, str] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            left, y_a, right, y_b = [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
        default_system = "top_left" if key == "bbox" else "bottom_left"
        return left, y_a, right, y_b, default_system

    if not isinstance(value, dict):
        return None

    try:
        if all(name in value for name in ("left", "bottom", "right", "top")):
            return (
                float(value["left"]),
                float(value["bottom"]),
                float(value["right"]),
                float(value["top"]),
                "bottom_left",
            )

        if all(name in value for name in ("l", "b", "r", "t")):
            return (
                float(value["l"]),
                float(value["b"]),
                float(value["r"]),
                float(value["t"]),
                "bottom_left",
            )

        if all(name in value for name in ("x0", "y0", "x1", "y1")):
            return (
                float(value["x0"]),
                float(value["y0"]),
                float(value["x1"]),
                float(value["y1"]),
                "top_left",
            )

        if all(name in value for name in ("x", "y", "width", "height")):
            x = float(value["x"])
            y = float(value["y"])
            return (x, y, x + float(value["width"]), y + float(value["height"]), "top_left")
    except (TypeError, ValueError):
        return None

    return None


def _bbox_coordinate_system(node: dict[str, Any], key: str, default: str) -> str:
    candidates = [
        node.get("bbox_coordinate_system"),
        node.get("coordinate_system"),
        node.get("coordinateSystem"),
        node.get("bbox_origin"),
        node.get("origin"),
    ]
    metadata = node.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("bbox_coordinate_system"),
                metadata.get("coordinate_system"),
                metadata.get("coordinateSystem"),
                metadata.get("bbox_origin"),
                metadata.get("origin"),
            ]
        )

    for value in candidates:
        normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
        if not normalized:
            continue
        if "bottom" in normalized:
            return "bottom_left"
        if "top" in normalized:
            return "top_left"

    return "top_left" if key == "bbox" else default


def _image_path(node: dict[str, Any], *, base_path: Path | None) -> str | None:
    for key in ("source", "image_path", "image path", "path", "file"):
        value = node.get(key)

        if isinstance(value, str) and value.strip():
            raw = value.strip()

            if raw.startswith("data:"):
                return raw

            path = Path(raw)

            if path.is_absolute() or base_path is None:
                return str(path)

            return str((base_path / path).resolve())

    return None


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "source": OpenDataLoaderExtractor.engine_name,
        "raw_type": str(node.get("type") or ""),
    }

    for key in (
        "font",
        "font size",
        "text color",
        "hidden text",
        "numbering style",
        "linked content id",
        "previous table id",
        "next table id",
        "previous list id",
        "next list id",
    ):
        if key in node:
            metadata[key.replace(" ", "_")] = node[key]

    if any(key in node for key in ("bounding box", "bounding_box", "boundingBox", "bbox")):
        bbox_key = next(key for key in ("bounding box", "bounding_box", "boundingBox", "bbox") if key in node)
        metadata["bbox_coordinate_system"] = _bbox_coordinate_system(node, bbox_key, "bottom_left")

    return metadata


def _detect_page_sizes(pdf_path: Path) -> dict[int, tuple[float, float]]:
    try:
        import fitz  # type: ignore
    except Exception:
        return {}

    try:
        with fitz.open(pdf_path) as doc:
            return {
                index: (float(page.rect.width), float(page.rect.height))
                for index, page in enumerate(doc, start=1)
            }
    except Exception as exc:
        logger.debug("Taille de pages indisponible pour OpenDataLoader: %s", exc)
        return {}
