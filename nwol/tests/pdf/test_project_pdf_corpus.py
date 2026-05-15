from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pdf.pipeline import clear_cache
from pdf.validation import PROJECT_PDF_NAMES, _block_covered_pages, validate_project_pdf_corpus


def test_project_pdf_corpus_validation_report():
    pytest.importorskip("fitz")
    project_root = Path(__file__).resolve().parents[3]
    missing = [name for name in PROJECT_PDF_NAMES if not (project_root / name).exists()]
    if missing:
        pytest.skip(f"PDF de validation absents: {missing}")

    clear_cache()
    reports = validate_project_pdf_corpus(project_root)

    assert {Path(report["pdf_path"]).name for report in reports} == set(PROJECT_PDF_NAMES)
    for report in reports:
        assert report["pages"] > 0, report
        assert report["blocks"] > 0, report
        assert report["pages_with_blocks"], report
        assert report["score"] >= 0.5, report
        assert not report["empty_text_blocks"], report
        assert not report["missing_assets"], report
        assert not report["formulas_without_render"], report


def test_validation_covered_pages_uses_page_spans():
    assert _block_covered_pages({"page_start": 5, "page_end": 6}) == [5, 6]
    assert _block_covered_pages({"page_number": 7}) == [7]
