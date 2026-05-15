from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .pipeline import build_document_model


PROJECT_PDF_NAMES = (
    "MetaRCNN.pdf",
    "Analyse_asymptotique_ameliore.pdf",
    "3DUNET.pdf",
    "1-spe-suites-numeriques.pdf",
)


def validate_project_pdf_corpus(project_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    return [build_pdf_validation_report(root / name) for name in PROJECT_PDF_NAMES]


def build_pdf_validation_report(pdf_path: str | Path) -> dict[str, Any]:
    path = Path(pdf_path)
    document = build_document_model(str(path), preferred_engine="auto")
    blocks = document.to_reader_blocks()
    block_types = Counter(str(block.get("type") or "unknown") for block in blocks)
    pages_with_blocks = sorted(
        page
        for block in blocks
        for page in _block_covered_pages(block)
    )
    pages_with_blocks = sorted(set(pages_with_blocks))

    return {
        "pdf_path": str(path),
        "exists": path.exists(),
        "engine": document.engine_name,
        "pages": document.pages,
        "pages_with_blocks": pages_with_blocks,
        "missing_pages": [
            page
            for page in range(1, int(document.pages or 0) + 1)
            if page not in pages_with_blocks
        ],
        "score": document.score,
        "warnings": list(document.warnings),
        "blocks": len(blocks),
        "block_types": dict(sorted(block_types.items())),
        "heading_levels": dict(sorted(_heading_levels(blocks).items())),
        "empty_text_blocks": _empty_text_blocks(blocks),
        "missing_assets": _missing_assets(blocks),
        "formula_blocks": block_types.get("formula", 0),
        "formulas_without_render": _formulas_without_render(blocks),
        "figure_blocks": block_types.get("figure", 0),
        "figures_without_image": _figures_without_image(blocks),
        "context_asset_blocks": sum(1 for block in blocks if _metadata(block).get("context_asset_path")),
        "display_context_asset_blocks": sum(1 for block in blocks if _metadata(block).get("context_asset_display")),
    }


def _heading_levels(blocks: list[dict[str, Any]]) -> Counter[int]:
    levels: Counter[int] = Counter()
    for block in blocks:
        if block.get("type") != "heading":
            continue
        try:
            levels[int(block.get("level") or 1)] += 1
        except (TypeError, ValueError):
            levels[1] += 1
    return levels


def _empty_text_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    empty: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.get("type") in {"figure"}:
            continue
        text = block.get("text") or block.get("latex") or block.get("markdown") or block.get("html") or ""
        if not str(text).strip() and _has_visual_render(block):
            continue
        if str(text).strip():
            continue
        empty.append(_block_ref(index, block))
    return empty


def _missing_assets(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        for field, path in _asset_paths(block):
            if path and not Path(path).exists():
                missing.append({**_block_ref(index, block), "field": field, "path": path})
    return missing


def _formulas_without_render(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.get("type") != "formula":
            continue
        metadata = _metadata(block)
        has_latex = bool(str(block.get("latex") or "").strip())
        has_image = bool(block.get("image_path") or metadata.get("formula_image_path"))
        can_dynamic_crop = metadata.get("render_mode") == "pdf_crop" and block.get("bbox") and _block_page(block)
        if not (has_latex or has_image or can_dynamic_crop):
            missing.append(_block_ref(index, block))
    return missing


def _figures_without_image(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.get("type") != "figure":
            continue
        if block.get("image_path"):
            continue
        if _metadata(block).get("context_asset_path"):
            continue
        missing.append(_block_ref(index, block))
    return missing


def _asset_paths(block: dict[str, Any]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if block.get("image_path"):
        paths.append(("image_path", str(block["image_path"])))
    metadata = _metadata(block)
    for key in ("formula_image_path", "context_asset_path", "table_image_path"):
        if metadata.get(key):
            paths.append((f"metadata.{key}", str(metadata[key])))
    for asset_index, asset in enumerate(metadata.get("llm_assets") or []):
        if isinstance(asset, dict) and asset.get("path"):
            paths.append((f"metadata.llm_assets[{asset_index}].path", str(asset["path"])))
    return paths


def _has_visual_render(block: dict[str, Any]) -> bool:
    metadata = _metadata(block)
    return bool(
        block.get("image_path")
        or metadata.get("formula_image_path")
        or metadata.get("table_image_path")
        or metadata.get("context_asset_path")
    )


def _block_ref(index: int, block: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "type": block.get("type"),
        "page": _block_page(block),
        "text": _short_text(block),
    }


def _short_text(block: dict[str, Any], limit: int = 120) -> str:
    text = str(block.get("latex") or block.get("text") or block.get("caption") or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _block_page(block: dict[str, Any]) -> int | None:
    for key in ("page_number", "page_start", "page"):
        try:
            value = block.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _block_page_end(block: dict[str, Any], fallback: int | None = None) -> int | None:
    for key in ("page_end", "page_number", "page_start", "page"):
        try:
            value = block.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return fallback


def _block_covered_pages(block: dict[str, Any]) -> list[int]:
    start = _block_page(block)
    if start is None:
        return []
    end = _block_page_end(block, start) or start
    if end < start:
        end = start
    return list(range(start, end + 1))


def _metadata(block: dict[str, Any]) -> dict[str, Any]:
    metadata = block.get("metadata")
    return metadata if isinstance(metadata, dict) else {}
