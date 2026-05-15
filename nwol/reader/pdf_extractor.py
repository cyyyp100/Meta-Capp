# reader/pdf_extractor.py — Compatibility wrapper around document/pdf_router.py
from __future__ import annotations

from pathlib import Path

try:
    from pdf.pipeline import build_document_model
except ModuleNotFoundError:
    from nwol.pdf.pipeline import build_document_model
try:
    from reader.pdf_result import PreprocessResult
except ModuleNotFoundError:
    from nwol.reader.pdf_result import PreprocessResult


class PDFExtractionError(RuntimeError):
    pass


class StructuredPDFExtractor:
    """
    Backward-compatible reader facade.

    The extraction intelligence now lives in document/. This class remains only
    for existing imports and examples that expect extract_blocks().
    """

    engine_name = "pymupdf_structured"

    def extract_blocks(self, pdf_path: str | Path) -> PreprocessResult:
        try:
            result = build_document_model(str(pdf_path), preferred_engine="auto")
        except Exception as exc:
            raise PDFExtractionError(f"Extraction PDF échouée: {exc}") from exc
        self.engine_name = result.engine_name
        return PreprocessResult(
            blocks=result.to_reader_blocks(),
            score=result.score,
            warnings=result.warnings,
            stats={
                "pages": result.pages,
                "blocks": len(result.blocks),
                "engine": result.engine_name,
                "debug_paths": list(result.debug_paths),
            },
        )


def extract_pdf_blocks(pdf_path: str | Path) -> PreprocessResult:
    return StructuredPDFExtractor().extract_blocks(pdf_path)
