from __future__ import annotations

import logging
from pathlib import Path

from document.extractors.base import (
    BaseExtractor,
    OptionalBackendUnavailable,
    markdown_to_document_blocks,
)
from document.models import DocumentBlock, ExtractionResult

logger = logging.getLogger("Document.marker")

# Séparateur de pages injecté par Marker dans le markdown de sortie
_PAGE_SEP = "-" * 48

# Modèles chargés une seule fois (lourd : ~2 Go, ~30 s)
_MODEL_CACHE: dict | None = None


def _parse_marker_pages(md_text: str, base_path: Path | None) -> list[DocumentBlock]:
    """Découpe le markdown Marker par page et assigne le numéro de page à chaque bloc."""
    raw_pages = md_text.split(_PAGE_SEP)
    all_blocks: list[DocumentBlock] = []
    for page_num, page_md in enumerate(raw_pages, start=1):
        page_md = page_md.strip()
        if not page_md:
            continue
        page_blocks = markdown_to_document_blocks(page_md, base_path=base_path)
        for block in page_blocks:
            block.page = page_num
        all_blocks.extend(page_blocks)
    return all_blocks


class MarkerExtractor(BaseExtractor):
    engine_name = "marker"

    def extract(self, pdf_path: str) -> ExtractionResult:
        global _MODEL_CACHE
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except ImportError as exc:
            raise OptionalBackendUnavailable(f"marker-pdf non installé : {exc}") from exc

        try:
            if _MODEL_CACHE is None:
                logger.info("Marker: chargement des modèles (première fois, ~30 s)…")
                _MODEL_CACHE = create_model_dict()

            converter = PdfConverter(artifact_dict=_MODEL_CACHE)
            rendered = converter(pdf_path)
            md_text = rendered.markdown

            try:
                import fitz
                doc = fitz.open(pdf_path)
                pages = doc.page_count
                doc.close()
            except Exception:
                pages = max(1, md_text.count(_PAGE_SEP) + 1)

            blocks = _parse_marker_pages(md_text, base_path=Path(pdf_path).parent)

            try:
                from document.postprocess.math_normalizer import normalize_math_blocks
                blocks = normalize_math_blocks(blocks)
            except Exception:
                pass

            result = ExtractionResult(
                blocks=blocks,
                pages=pages,
                score=0.0,
                warnings=[],
                engine_name="marker",
                debug_paths=[],
            )

            try:
                from document.postprocess.quality import update_result_quality
                update_result_quality(result)
            except Exception:
                pass

            return result

        except OptionalBackendUnavailable:
            raise
        except Exception as exc:
            msg = str(exc)
            if "meta tensor" in msg.lower() or "Cannot copy out of meta tensor" in msg:
                raise OptionalBackendUnavailable(
                    f"Marker: erreur torch meta tensor — essayez de mettre à jour PyTorch : {exc}"
                ) from exc
            raise
