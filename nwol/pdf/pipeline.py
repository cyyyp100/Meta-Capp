from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ._legacy import ensure_legacy_document_path
from .model import BBox, DocumentBlock, DocumentModel

ensure_legacy_document_path()

from document.pdf_router import (  # noqa: E402
    clear_cache as clear_document_router_cache,
    compare_pdf_backends as compare_document_backends,
    extract_document,
)

logger = logging.getLogger("PDF.pipeline")

_result_cache: OrderedDict[tuple[str, float, str], DocumentModel] = OrderedDict()
_CACHE_MAX_SIZE = 8


def build_document_model(pdf_path: str, preferred_engine: str = "auto") -> DocumentModel:
    path = str(Path(pdf_path).resolve())
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    key = (path, mtime, preferred_engine)
    if key in _result_cache:
        _result_cache.move_to_end(key)
        return _result_cache[key]

    document = _run_extraction(path, preferred_engine)
    if len(_result_cache) >= _CACHE_MAX_SIZE:
        _result_cache.popitem(last=False)
    _result_cache[key] = document
    return document


def clear_cache() -> None:
    _result_cache.clear()
    clear_document_router_cache()


def _run_extraction(pdf_path: str, preferred_engine: str = "auto") -> DocumentModel:
    result = extract_document(pdf_path, preferred_engine=preferred_engine)
    blocks = [_from_document_block(block) for block in result.blocks]
    _tag_blocks(blocks, result.engine_name)
    return DocumentModel(
        blocks=blocks,
        pages=result.pages,
        score=result.score,
        warnings=list(result.warnings),
        engine_name=result.engine_name,
        debug_paths=list(result.debug_paths),
    )


def _tag_blocks(blocks: list[DocumentBlock], engine_name: str) -> None:
    for index, block in enumerate(blocks):
        block.metadata.setdefault("engine", engine_name)
        block.metadata.setdefault("block_index", index)


def compare_pdf_backends(pdf_path: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    return compare_document_backends(pdf_path, output_dir=output_dir)


def _from_document_block(block: object) -> DocumentBlock:
    bbox = _convert_bbox(getattr(block, "bbox", None))
    metadata = dict(getattr(block, "metadata", None) or {})
    return DocumentBlock(
        type=str(getattr(block, "type", None) or "paragraph"),
        text=str(getattr(block, "text", None) or ""),
        page=getattr(block, "page", None),
        bbox=bbox,
        level=getattr(block, "level", None),
        items=list(getattr(block, "items", None) or []) if getattr(block, "items", None) is not None else None,
        latex=getattr(block, "latex", None),
        html=getattr(block, "html", None),
        markdown=getattr(block, "markdown", None),
        image_path=getattr(block, "image_path", None),
        caption=getattr(block, "caption", None),
        confidence=float(getattr(block, "confidence", 1.0) or 1.0),
        metadata=metadata,
        id=getattr(block, "id", None),
    )


def _convert_bbox(bbox: object) -> BBox | None:
    if bbox is None:
        return None
    if isinstance(bbox, BBox):
        return bbox
    if hasattr(bbox, "to_list"):
        return BBox.from_seq(bbox.to_list())
    return BBox.from_seq(bbox)
