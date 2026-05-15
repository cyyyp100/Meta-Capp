from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pdf.model import BBox, DocumentBlock, DocumentModel


def test_to_reader_dict_paragraph_aliases_and_metadata_flattening():
    block = DocumentBlock(
        type="paragraph",
        text="Texte",
        page=2,
        bbox=BBox(1, 2, 3, 4),
        confidence=1.5,
        metadata={
            "engine": "pymupdf_structured",
            "block_index": 7,
            "is_caption": True,
            "is_metadata": False,
            "caption_display": "Figure 1",
            "caption_group": "g1",
        },
        id="p2_b7",
    )

    assert block.to_reader_dict() == {
        "type": "paragraph",
        "id": "p2_b7",
        "text": "Texte",
        "page_number": 2,
        "page_start": 2,
        "page_end": 2,
        "bbox": [1, 2, 3, 4],
        "confidence": 1.0,
        "metadata": {
            "engine": "pymupdf_structured",
            "block_index": 7,
            "is_caption": True,
            "is_metadata": False,
            "caption_display": "Figure 1",
            "caption_group": "g1",
        },
        "is_caption": True,
        "is_metadata": False,
        "engine": "pymupdf_structured",
        "block_index": 7,
        "caption_display": "Figure 1",
        "caption_group": "g1",
    }


def test_to_reader_dict_each_block_shape():
    blocks = [
        DocumentBlock(type="heading", text="Titre", level=2),
        DocumentBlock(type="formula", latex=r"x^2", confidence=-1),
        DocumentBlock(type="bullet_list", items=["a", "b"]),
        DocumentBlock(type="table", markdown="| a |", html="<table></table>"),
        DocumentBlock(type="figure", image_path="/tmp/f.png", caption="Figure 1"),
        DocumentBlock(type="code", text="print(1)"),
    ]
    reader = DocumentModel(blocks=blocks, pages=1, score=1.0, warnings=[], engine_name="test").to_reader_blocks()

    assert reader[0]["level"] == 2
    assert reader[1]["text"] == r"x^2"
    assert reader[1]["latex"] == r"x^2"
    assert reader[1]["display"] is True
    assert reader[1]["confidence"] == 0.0
    assert reader[2]["text"] == "• a\n• b"
    assert reader[2]["items"] == ["a", "b"]
    assert reader[3]["markdown"] == "| a |"
    assert reader[3]["html"] == "<table></table>"
    assert reader[4]["caption"] == "Figure 1"
    assert reader[4]["text"] == "Figure 1"
    assert reader[5]["text"] == "print(1)"


def test_to_reader_dict_preserves_metadata_page_span():
    block = DocumentBlock(
        type="paragraph",
        text="Paragraphe qui continue sur la page suivante.",
        page=5,
        metadata={"page_start": 5, "page_end": 6},
    )

    reader = block.to_reader_dict()

    assert reader["page_number"] == 5
    assert reader["page_start"] == 5
    assert reader["page_end"] == 6


def test_from_reader_dict_round_trip_keeps_extra_keys_as_metadata():
    original = {
        "type": "paragraph",
        "text": "Bonjour",
        "page_number": "3",
        "bbox": [0, 1, 2, 3],
        "confidence": 0.8,
        "engine": "legacy",
        "block_index": 1,
    }

    block = DocumentBlock.from_reader_dict(original)

    assert block.page == 3
    assert block.bbox == BBox(0, 1, 2, 3)
    assert block.metadata["engine"] == "legacy"
    assert block.to_reader_dict()["page_start"] == 3


def test_from_reader_dict_round_trip_keeps_page_span():
    original = {
        "type": "paragraph",
        "text": "Paragraphe sur deux pages.",
        "page_start": 3,
        "page_end": 4,
    }

    block = DocumentBlock.from_reader_dict(original)

    assert block.page == 3
    assert block.metadata["page_start"] == 3
    assert block.metadata["page_end"] == 4
    assert block.to_reader_dict()["page_end"] == 4
