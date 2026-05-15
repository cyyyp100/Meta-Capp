from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.document import normalize_chapter_list
from document.extractors.base import OptionalBackendUnavailable
from document.extractors.opendataloader_extractor import OpenDataLoaderExtractor
from document.models import DocumentBlock
from document.postprocess.learning_chunks import detect_document_type

logger = logging.getLogger("Document.global_index")


@dataclass(slots=True)
class DocumentGlobalIndex:
    doc_id: int
    pdf_path: str
    pdf_hash: str
    page_count: int
    chapters: list[dict] = field(default_factory=list)
    assets: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    headings: list[dict] = field(default_factory=list)
    document_type: str = "book"
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_backend_report(self, *, status: str, engine_name: str, debug_paths: list[str]) -> dict[str, Any]:
        return {
            "status": status,
            "engine": engine_name,
            "page_count": self.page_count,
            "score": self.score,
            "warnings": list(self.warnings),
            "debug_paths": list(debug_paths),
            "headings": len(self.headings),
            "assets": len(self.assets),
            "tables": len(self.tables),
        }


def build_document_global_index(doc_id: int, pdf_path: str) -> tuple[DocumentGlobalIndex, str, dict[str, Any]]:
    """Run OpenDataLoader as a global indexer, never as the final page reader."""
    path = Path(pdf_path)
    pdf_hash = hash_pdf(path)
    page_count = _page_count(path)
    status = "complete"
    debug_paths: list[str] = []
    engine_name = OpenDataLoaderExtractor.engine_name

    try:
        result = OpenDataLoaderExtractor().extract(str(path))
        blocks = result.blocks
        page_count = result.pages or page_count
        warnings = list(result.warnings)
        score = float(result.score or 0.0)
        debug_paths = list(result.debug_paths)
        document_type = detect_document_type(blocks, pages=page_count)
    except OptionalBackendUnavailable as exc:
        blocks = []
        warnings = [str(exc)]
        score = 0.0
        status = "unavailable"
        document_type = "book"
        logger.info("[ODL_INDEX] OpenDataLoader indisponible pour %s: %s", path.name, exc)
    except Exception as exc:
        blocks = []
        warnings = [f"OpenDataLoader a échoué: {exc}"]
        score = 0.0
        status = "failed"
        document_type = "book"
        logger.warning("[ODL_INDEX] échec index global %s: %s", path.name, exc)

    headings = _headings(blocks)
    chapters = normalize_chapter_list(_chapters_from_headings(headings), page_count) if headings else []
    assets = _assets(blocks)
    tables = _tables(blocks)

    index = DocumentGlobalIndex(
        doc_id=doc_id,
        pdf_path=str(path),
        pdf_hash=pdf_hash,
        page_count=page_count,
        chapters=chapters,
        assets=assets,
        tables=tables,
        headings=headings,
        document_type=document_type,
        warnings=warnings,
        score=score,
    )
    return index, status, index.to_backend_report(status=status, engine_name=engine_name, debug_paths=debug_paths)


def hash_pdf(pdf_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_count(path: Path) -> int:
    try:
        import fitz  # type: ignore

        with fitz.open(path) as doc:
            return len(doc)
    except Exception:
        return 0


def _headings(blocks: list[DocumentBlock]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.type != "heading":
            continue
        result.append(
            {
                "id": block.id or f"h{index}",
                "title": (block.text or "").strip(),
                "page_start": int(block.page or 1),
                "toc_level": int(block.level or 1),
                "bbox": block.bbox.to_list() if block.bbox else None,
                "source": block.metadata.get("source") or block.metadata.get("engine") or OpenDataLoaderExtractor.engine_name,
            }
        )
    return [heading for heading in result if heading["title"]]


def _chapters_from_headings(headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": heading["title"],
            "page_start": heading["page_start"],
            "toc_level": heading["toc_level"],
        }
        for heading in headings
    ]


def _assets(blocks: list[DocumentBlock]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.type != "figure":
            continue
        result.append(
            {
                "id": block.id or f"asset{index}",
                "type": "figure",
                "page": int(block.page or 1),
                "bbox": block.bbox.to_list() if block.bbox else None,
                "image_path": block.image_path,
                "caption": block.caption or block.text or "",
                "source": block.metadata.get("source") or block.metadata.get("engine") or OpenDataLoaderExtractor.engine_name,
                "confidence": float(block.confidence or 0.0),
            }
        )
    return result


def _tables(blocks: list[DocumentBlock]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if block.type != "table":
            continue
        result.append(
            {
                "id": block.id or f"table{index}",
                "page": int(block.page or 1),
                "bbox": block.bbox.to_list() if block.bbox else None,
                "markdown": block.markdown,
                "html": block.html,
                "table_image_path": (block.metadata or {}).get("table_image_path"),
                "source": block.metadata.get("source") or block.metadata.get("engine") or OpenDataLoaderExtractor.engine_name,
                "confidence": float(block.confidence or 0.0),
            }
        )
    return result
