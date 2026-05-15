from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from core.scopes import TextScope
from document.extractors.opendataloader_extractor import OpenDataLoaderExtractor
from document.extractors.pymupdf_extractor import PyMuPDFExtractor
from document.models import DocumentBlock as RouterBlock
from document.models import ExtractionResult
from pdf.pipeline import build_document_model, clear_cache
from document.extractors.opendataloader_extractor import convert_opendataloader_json_to_document_blocks


def test_pipeline_smoke_and_textscope_compatibility(tmp_path):
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapitre 1", fontsize=20)
    page.insert_text(
        (72, 120),
        "Ce paragraphe de cours contient assez de texte pour etre extrait comme paragraphe.",
        fontsize=11,
    )
    doc.save(pdf_path)
    doc.close()
    clear_cache()

    document = build_document_model(str(pdf_path))
    blocks = document.to_reader_blocks()
    scope = TextScope("document", "Document complet", 1, document.pages, blocks)

    assert document.pages == 1
    assert blocks
    assert all("type" in block and "confidence" in block for block in blocks)
    assert scope.plain_text().strip()


def test_extract_page_lazy_returns_page_plan_and_reader_blocks(tmp_path):
    fitz = pytest.importorskip("fitz")
    from core.parser import extract_page_lazy

    pdf_path = tmp_path / "lazy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapitre 1", fontsize=20)
    page.insert_text((72, 120), "Un paragraphe lisible pour la lecture page par page.", fontsize=11)
    doc.save(pdf_path)
    doc.close()

    result = extract_page_lazy(str(pdf_path), 1, enrich_assets=False)
    reader_blocks = result.to_reader_blocks()

    assert reader_blocks
    assert result.page_plan.reading_order_ids
    assert result.layout_risk.score >= 0.0
    assert all(block.get("id") for block in reader_blocks)


def test_extract_page_lazy_wrapper_accepts_llm_controls(tmp_path):
    fitz = pytest.importorskip("fitz")
    from core.parser import extract_page_lazy

    pdf_path = tmp_path / "lazy_no_llm.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Introduction", fontsize=20)
    page.insert_text((72, 120), "Un paragraphe court pour tester le wrapper.", fontsize=11)
    doc.save(pdf_path)
    doc.close()

    result = extract_page_lazy(
        str(pdf_path),
        1,
        enrich_assets=False,
        validate_with_llm=False,
        llm_generation=123,
    )

    assert result.to_reader_blocks()


def test_extract_page_lazy_filters_repeated_pdf_headers(tmp_path):
    fitz = pytest.importorskip("fitz")
    from core.parser import extract_page_lazy

    pdf_path = tmp_path / "headers.pdf"
    doc = fitz.open()
    for page_number in range(1, 5):
        page = doc.new_page()
        page.insert_text((72, 28), "Published as a conference paper at ICLR 2019", fontsize=10)
        page.insert_text((72, 120), f"4.{page_number} Section body text that should remain visible.", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    result = extract_page_lazy(str(pdf_path), 3, enrich_assets=False, validate_with_llm=False)
    text = "\n".join(block.text for block in result.blocks if block.text)

    assert "Published as a conference paper" not in text
    assert "4.3 Section body text" in text


def test_opendataloader_json_maps_to_reader_contract():
    data = {
        "number of pages": 1,
        "kids": [
            {
                "type": "heading",
                "content": "Introduction",
                "heading level": 1,
                "page number": 1,
                "bounding box": [72, 700, 260, 724],
            },
            {
                "type": "paragraph",
                "content": "Un paragraphe exploitable par le reader.",
                "page number": 1,
                "bounding box": [72, 650, 480, 680],
            },
            {
                "type": "table",
                "page number": 1,
                "bounding box": [72, 500, 360, 620],
                "rows": [["Nom", "Valeur"], ["a", "1"]],
            },
        ],
    }

    blocks = convert_opendataloader_json_to_document_blocks(data, page_sizes={1: (595.0, 842.0)})
    reader_blocks = [block.to_reader_dict() for block in blocks]

    assert [block["type"] for block in reader_blocks] == ["heading", "paragraph", "table"]
    assert all("type" in block and "text" in block for block in reader_blocks)
    assert reader_blocks[0]["bbox"] == [72.0, 118.0, 260.0, 142.0]
    assert "| Nom | Valeur |" in reader_blocks[2]["markdown"]


def test_opendataloader_json_nested_variants_are_normalized():
    data = {
        "pages": [
            {
                "type": "page",
                "page": 1,
                "children": [
                    {
                        "type": "paragraph",
                        "content": {"spans": [{"text": "Texte"}, {"text": "imbriqué"}]},
                        "bbox": {"x0": 10, "y0": 20, "x1": 160, "y1": 42},
                    },
                    {
                        "type": "table",
                        "bbox": {"x": 10, "y": 70, "width": 180, "height": 80},
                        "cells": [
                            {"row": 0, "column": 0, "text": "Nom"},
                            {"row": 0, "column": 1, "text": "Valeur"},
                            {"row": 1, "column": 0, "text": "alpha"},
                            {"row": 1, "column": 1, "text": "1"},
                        ],
                    },
                ],
            }
        ]
    }

    blocks = convert_opendataloader_json_to_document_blocks(data, page_sizes={1: (595.0, 842.0)})
    reader_blocks = [block.to_reader_dict() for block in blocks]

    assert reader_blocks[0]["text"] == "Texte imbriqué"
    assert reader_blocks[0]["bbox"] == [10.0, 20.0, 160.0, 42.0]
    assert reader_blocks[1]["bbox"] == [10.0, 70.0, 190.0, 150.0]
    assert "| alpha | 1 |" in reader_blocks[1]["markdown"]


def test_opendataloader_structural_list_sections_preserve_child_paragraphs():
    data = {
        "number of pages": 1,
        "kids": [
            {
                "type": "list",
                "page number": 1,
                "bounding box": [72, 300, 500, 700],
                "list items": [
                    {
                        "type": "list item",
                        "content": "4.2 EXPERIMENTS",
                        "page number": 1,
                        "bounding box": [72, 650, 180, 666],
                        "kids": [
                            {
                                "type": "paragraph",
                                "content": "To evaluate our methods we used a hierarchical search.",
                                "page number": 1,
                                "bounding box": [72, 610, 500, 630],
                            },
                            {
                                "type": "paragraph",
                                "content": "An experiment consisted of training for 150 epochs.",
                                "page number": 1,
                                "bounding box": [72, 580, 500, 600],
                            },
                        ],
                    },
                    {
                        "type": "list item",
                        "content": "4.3 RESULTS Our proposed methodologies improve MAML.",
                        "page number": 1,
                        "bounding box": [72, 540, 500, 560],
                    },
                ],
            }
        ],
    }

    blocks = convert_opendataloader_json_to_document_blocks(data, page_sizes={1: (595.0, 842.0)})

    assert [block.type for block in blocks] == ["paragraph", "paragraph", "paragraph", "paragraph"]
    assert [block.text for block in blocks] == [
        "4.2 EXPERIMENTS",
        "To evaluate our methods we used a hierarchical search.",
        "An experiment consisted of training for 150 epochs.",
        "4.3 RESULTS Our proposed methodologies improve MAML.",
    ]


def test_document_block_round_trip_preserves_page_span():
    block = RouterBlock.from_reader_dict(
        {
            "type": "paragraph",
            "text": "Paragraphe sur deux pages.",
            "page_start": 3,
            "page_end": 4,
        }
    )

    reader = block.to_reader_dict()

    assert reader["page_start"] == 3
    assert reader["page_end"] == 4


def test_opendataloader_fallback_to_pymupdf(monkeypatch):
    clear_cache()

    def fail_opendataloader(self, pdf_path):
        raise RuntimeError("boom")

    def fake_pymupdf_extract(self, pdf_path):
        return ExtractionResult(
            blocks=[RouterBlock(type="paragraph", text="Texte de fallback", page=1)],
            pages=1,
            score=0.9,
            warnings=[],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    monkeypatch.setattr(OpenDataLoaderExtractor, "extract", fail_opendataloader)
    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf_extract)

    document = build_document_model("/tmp/missing.pdf", preferred_engine="auto")

    assert document.blocks
    assert document.engine_name == "pymupdf_structured"
    clear_cache()


def test_reader_does_not_import_legacy_preprocessor():
    source = (Path(__file__).parents[2] / "reader" / "pdf_extractor.py").read_text(encoding="utf-8")

    assert "pdf_preprocessor" not in source
