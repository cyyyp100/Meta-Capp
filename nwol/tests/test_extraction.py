# tests/test_extraction.py
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import parser
from core.parser import _parse_markdown_blocks, _sort_pdf_blocks_for_reading


def test_markdown_parser():
    blocks = _parse_markdown_blocks(
        """# Chapitre 1

Un paragraphe avec $x^2$ en formule inline.

$$
E=mc^2
$$

```python
def f(x):
    return x * x
```

![Schema](images/schema.png)
Figure 1: Energie et masse
""",
        base_path="/tmp/course",
    )

    assert blocks[0] == {"type": "heading", "level": 1, "text": "Chapitre 1"}
    assert blocks[1]["type"] == "paragraph"
    assert blocks[1]["inline_formulas"] == ["x^2"]
    assert blocks[2]["type"] == "formula"
    assert blocks[2]["display"] is True
    assert blocks[2]["latex"] == "E=mc^2"
    assert blocks[3]["type"] == "code"
    assert "return x * x" in blocks[3]["text"]
    assert blocks[4]["type"] == "figure"
    assert blocks[4]["image_path"].endswith("/tmp/course/images/schema.png")
    assert blocks[4]["caption"] == "Figure 1: Energie et masse"


def test_formula_detection():
    blocks = _parse_markdown_blocks("$$E=mc^2$$")

    assert len(blocks) == 1
    assert blocks[0]["type"] == "formula"
    assert blocks[0]["display"] is True
    assert blocks[0]["latex"] == "E=mc^2"


def test_multicolumn_order():
    raw_blocks = [
        {"id": "right-2", "bbox": [320, 140, 560, 160]},
        {"id": "left-2", "bbox": [60, 140, 290, 160]},
        {"id": "right-1", "bbox": [320, 80, 560, 100]},
        {"id": "left-1", "bbox": [60, 80, 290, 100]},
    ]

    ordered = _sort_pdf_blocks_for_reading(raw_blocks, page_width=620)

    assert [block["id"] for block in ordered] == [
        "left-1",
        "left-2",
        "right-1",
        "right-2",
    ]


def test_engine_cascade(monkeypatch):
    def fake_available(engine):
        return engine == "pymupdf"

    monkeypatch.setattr(parser, "_engine_available", fake_available)

    assert parser.detect_best_engine() == "pymupdf"
    assert parser._fallback_chain("unknown") == ("pymupdf",)


def test_detect_best_engine_prefers_structured(monkeypatch):
    def fake_available(engine):
        return engine in {parser.STRUCTURED_ENGINE, "pymupdf"}

    monkeypatch.setattr(parser, "_engine_available", fake_available)

    assert parser.detect_best_engine() == parser.STRUCTURED_ENGINE


def test_pymupdf_blocks(tmp_path):
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

    blocks = parser.extract_page(str(pdf_path), 1, "pymupdf")

    assert any(block.get("type") == "heading" for block in blocks)
    assert any(block.get("type") == "paragraph" for block in blocks)
