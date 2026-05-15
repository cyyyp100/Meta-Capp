from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from document.layout.page_reading_plan import build_page_reading_plan, reorder_blocks_by_ids
from document.layout import llm_geometry_validator
from document.layout.risk import compute_crop_risk, compute_layout_risk
from document.models import BoundingBox, DocumentBlock
from llm.pdf_assistant_queue import validate_reading_order_response


def _block(
    block_id: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    btype: str = "paragraph",
    text: str = "Texte de test",
) -> DocumentBlock:
    return DocumentBlock(
        type=btype,
        text=text,
        page=1,
        bbox=BoundingBox(x0, y0, x1, y1),
        id=block_id,
        metadata={"page_width": 600.0, "page_height": 800.0},
    )


def test_reorder_blocks_by_ids_preserves_unmentioned_blocks_at_end():
    a = _block("a", 50, 80, 250, 100)
    b = _block("b", 50, 120, 250, 140)
    c = _block("c", 50, 160, 250, 180)

    ordered = reorder_blocks_by_ids([a, b, c], ["c", "a"])

    assert [block.id for block in ordered] == ["c", "a", "b"]


def test_layout_risk_flags_two_columns_and_visual_anchors():
    blocks = [
        _block("l1", 50, 80, 250, 100),
        _block("l2", 50, 130, 250, 150),
        _block("r1", 340, 80, 560, 100),
        _block("r2", 340, 130, 560, 150),
        _block("fig", 60, 210, 540, 330, btype="figure", text="Figure 1."),
        _block("tbl", 60, 360, 540, 430, btype="table", text="A | B"),
    ]
    plan = build_page_reading_plan(1, blocks, page_width=600.0)

    risk = compute_layout_risk(blocks, plan, ["Layout deux colonnes détecté"], "scientific_article")

    assert risk.score >= 0.45
    assert risk.needs_llm_order is True
    assert "two_columns" in risk.reasons
    assert "many_visual_anchors" in risk.reasons


def test_crop_risk_flags_vector_figure_without_caption_near_caption():
    figure = _block("fig", 60, 100, 560, 260, btype="figure", text="")
    figure.metadata["source"] = "vector_graphic_drawing"
    caption = _block("cap", 70, 275, 550, 300, text="Figure 1. Un schéma.")

    risk = compute_crop_risk(figure, [figure, caption])

    assert risk.needs_llm is True
    assert "vector_reconstructed" in risk.reasons
    assert "near_caption_unattached" in risk.reasons


def test_llm_reading_order_response_must_be_exact_permutation():
    valid = ["a", "b", "c"]

    assert validate_reading_order_response(["b", "a", "c"], valid) == ["b", "a", "c"]
    assert validate_reading_order_response(["b", "a"], valid) is None
    assert validate_reading_order_response(["b", "a", "x"], valid) is None
    assert validate_reading_order_response(["b", "a", "a"], valid) is None


def test_llm_geometry_fallback_preserves_page_plan_order(monkeypatch):
    monkeypatch.setattr(llm_geometry_validator, "_call_ollama", lambda *args, **kwargs: "not json")
    plan_order = ["left-heading", "left-body", "right-heading", "right-body"]
    summaries = [
        {"id": "right-heading", "bbox": [310, 360, 455, 376]},
        {"id": "right-body", "bbox": [310, 390, 545, 440]},
        {"id": "left-heading", "bbox": [50, 640, 240, 656]},
        {"id": "left-body", "bbox": [50, 666, 286, 702]},
    ]

    result = llm_geometry_validator.validate_reading_order(plan_order, summaries)

    assert result == plan_order
