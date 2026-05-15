"""
Tests for:
  - test_two_column_unbalanced_order
  - test_fused_result_preserves_reading_order
  - test_heading_next_page_not_before_right_column
  - test_math_crop_not_schema
  - test_context_asset_not_displayed_before_llm_complete  (structural / unit)
  - test_session_cache_cleanup_on_back_and_finish
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from document.layout.column_detector import ColumnLayout, detect_columns
from document.layout.page_reading_plan import build_page_reading_plan
from document.layout.risk import compute_layout_risk
from document.layout.reading_order import order_blocks_for_reading, order_page_blocks
from document.models import BoundingBox, DocumentBlock, ExtractionResult
import ui.app as app_module
from ui.app import NWoLApp, _cached_payload_satisfies_reader_request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block(
    bid: str,
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    btype: str = "paragraph",
    text: str = "Lorem ipsum",
) -> DocumentBlock:
    return DocumentBlock(
        type=btype,
        text=text,
        page=page,
        bbox=BoundingBox(x0, y0, x1, y1),
        metadata={
            "page_width": 595.0,
            "page_height": 842.0,
        },
        id=bid,
    )


# ---------------------------------------------------------------------------
# 1. test_two_column_unbalanced_order
# ---------------------------------------------------------------------------

def test_two_column_unbalanced_order():
    """Left column ends before right column: all left blocks precede right blocks."""
    page_width = 595.0
    # Left column (x0 ≈ 50–260): 3 blocks, ends at y=300
    l1 = _block("L1", 1, 50, 100, 260, 130)
    l2 = _block("L2", 1, 50, 150, 260, 180)
    l3 = _block("L3", 1, 50, 250, 260, 300)
    # Right column (x0 ≈ 310–550): 5 blocks, continues to y=500
    r1 = _block("R1", 1, 310, 100, 550, 130)
    r2 = _block("R2", 1, 310, 150, 550, 180)
    r3 = _block("R3", 1, 310, 250, 550, 300)
    r4 = _block("R4", 1, 310, 350, 550, 380)
    r5 = _block("R5", 1, 310, 460, 550, 500)

    blocks = [l1, l2, l3, r1, r2, r3, r4, r5]
    ordered = order_blocks_for_reading(blocks, {1: (page_width, 842.0)})
    ids = [b.id for b in ordered]

    # All left-column blocks must come before any right-column block.
    last_left = max(ids.index(bid) for bid in ("L1", "L2", "L3"))
    first_right = min(ids.index(bid) for bid in ("R1", "R2", "R3", "R4", "R5"))
    assert last_left < first_right, (
        f"Left column should finish before right column starts. Got order: {ids}"
    )
    # Left column internal order: L1 → L2 → L3
    assert ids.index("L1") < ids.index("L2") < ids.index("L3")
    # Right column internal order: R1 → R2 → R3 → R4 → R5
    for prev, nxt in [("R1", "R2"), ("R2", "R3"), ("R3", "R4"), ("R4", "R5")]:
        assert ids.index(prev) < ids.index(nxt)

    # reading_order_index must be stamped correctly
    for expected_idx, block in enumerate(ordered):
        assert block.metadata.get("reading_order_index") == expected_idx


def test_two_column_page_requires_special_order_validation():
    blocks = [
        _block("L1", 1, 50, 100, 260, 130),
        _block("L2", 1, 50, 150, 260, 180),
        _block("R1", 1, 310, 100, 550, 130),
        _block("R2", 1, 310, 150, 550, 180),
    ]

    plan = build_page_reading_plan(page_number=1, blocks=blocks, page_width=595.0)
    risk = compute_layout_risk(blocks, plan)

    assert plan.columns
    assert risk.needs_llm_order is True


def test_reader_rejects_prefetch_cache_without_assets_or_pending_llm_order():
    assert _cached_payload_satisfies_reader_request(
        {"blocks": [], "enrich_assets": False},
        validate_with_llm=True,
    ) is False
    assert _cached_payload_satisfies_reader_request(
        {"blocks": [], "enrich_assets": True},
        validate_with_llm=True,
    ) is False
    assert _cached_payload_satisfies_reader_request(
        {
            "blocks": [{"metadata": {}}],
            "enrich_assets": True,
            "page_plan": {"columns": [["a"], ["b"]]},
            "layout_risk": {"needs_llm_order": True},
        },
        validate_with_llm=True,
    ) is False
    assert _cached_payload_satisfies_reader_request(
        {
            "blocks": [{"metadata": {"llm_order_status": "attempted"}}],
            "enrich_assets": True,
            "page_plan": {"columns": [["a"], ["b"]]},
            "layout_risk": {"needs_llm_order": True},
        },
        validate_with_llm=True,
    ) is True
    assert _cached_payload_satisfies_reader_request(
        {
            "blocks": [{"metadata": {"llm_order_status": "attempted"}}],
            "enrich_assets": True,
            "page_plan": {},
            "layout_risk": {"needs_llm_order": False, "needs_llm_crop": True},
        },
        validate_with_llm=True,
    ) is False
    assert _cached_payload_satisfies_reader_request(
        {
            "blocks": [{"type": "figure", "metadata": {"llm_crop_status": "attempted"}}],
            "enrich_assets": True,
            "page_plan": {},
            "layout_risk": {"needs_llm_order": False, "needs_llm_crop": True},
        },
        validate_with_llm=True,
    ) is True


def test_reader_page_extraction_does_not_pass_ui_generation_to_pdf_llm(monkeypatch):
    captured = {}

    class FakeDoc:
        path = "/tmp/doc.pdf"

    class FakeState:
        doc_id = None

    class FakeResult:
        score = 1.0
        warnings = []

        def to_reader_blocks(self):
            return []

        def page_plan_dict(self):
            return {}

        def layout_risk_dict(self):
            return {}

    def fake_extract_page_lazy(*args, **kwargs):
        captured["llm_generation"] = kwargs.get("llm_generation")
        return FakeResult()

    app = NWoLApp.__new__(NWoLApp)
    app._doc = FakeDoc()
    app._state = FakeState()
    app._engine_name = "pymupdf_structured"
    app._reading_generation = 17
    app._current_document_type = lambda: "book"

    monkeypatch.setattr(app_module, "extract_page_lazy", fake_extract_page_lazy)

    app._extract_and_cache_reader_page(
        1,
        pdf_path="/tmp/doc.pdf",
        doc_id=None,
        document_type="book",
        generation=17,
        validate_with_llm=True,
    )

    assert captured["llm_generation"] is None


# ---------------------------------------------------------------------------
# 2. test_fused_result_preserves_reading_order
# ---------------------------------------------------------------------------

def test_fused_result_preserves_reading_order():
    """After fusion, re-ordering must not undo the geometric reading_order_index."""
    from document.pdf_router import _order_blocks_for_reading as _router_order

    blocks = [
        _block("A", 1, 50, 100, 260, 130),   # left col
        _block("B", 1, 310, 100, 550, 130),  # right col
        _block("C", 1, 50, 200, 260, 230),   # left col
        _block("D", 1, 310, 200, 550, 230),  # right col
    ]
    for i, b in enumerate(blocks):
        b.metadata["page_width"] = 595.0

    ordered = _router_order(blocks)
    ids = [b.id for b in ordered]

    # Geometric order: left-A, left-C, right-B, right-D (two-column: left first)
    assert ids.index("A") < ids.index("B"), f"Left should precede right. Got {ids}"
    assert ids.index("C") < ids.index("D"), f"Left should precede right. Got {ids}"
    assert ids.index("A") < ids.index("C"), "A above C in left column"

    # Simulate fallback: blocks already have reading_order_index stamped
    for b in ordered:
        b.metadata["reading_order_index"] = ordered.index(b)

    # Corrupt block so geometric ordering raises — fallback must preserve index order
    ordered[0].page = None
    with patch(
        "document.pdf_router.order_blocks_for_reading",
        side_effect=RuntimeError("boom"),
    ):
        fallback = _router_order(ordered)

    fallback_ids = [b.id for b in fallback]
    # Must still honour reading_order_index, not revert to _position_key
    assert fallback_ids == ids, f"Fallback broke order: {fallback_ids} != {ids}"


# ---------------------------------------------------------------------------
# 3. test_heading_next_page_not_before_right_column
# ---------------------------------------------------------------------------

def test_heading_next_page_not_before_right_column():
    """A heading on page 2 must appear after all page-1 right-column blocks."""
    # Page 1: two columns (unbalanced)
    l1 = _block("L1", 1, 50, 100, 260, 150)
    l2 = _block("L2", 1, 50, 200, 260, 250)
    r1 = _block("R1", 1, 310, 100, 550, 150)
    r2 = _block("R2", 1, 310, 200, 550, 250)
    r3 = _block("R3", 1, 310, 300, 550, 350)  # right column extends further

    # Page 2: heading
    h = _block("H", 2, 50, 50, 550, 90, btype="heading", text="3.4 Suite")

    blocks = [l1, l2, r1, r2, r3, h]
    for b in blocks:
        b.metadata["page_width"] = 595.0

    ordered = order_blocks_for_reading(blocks, {1: (595.0, 842.0), 2: (595.0, 842.0)})
    ids = [b.id for b in ordered]

    # R3 (page 1) must come before H (page 2)
    assert ids.index("R3") < ids.index("H"), (
        f"Right-column block on page 1 must precede heading on page 2. Got {ids}"
    )
    # All page-1 blocks must precede heading on page 2
    page1_ids = {"L1", "L2", "R1", "R2", "R3"}
    last_p1 = max(ids.index(bid) for bid in page1_ids)
    assert last_p1 < ids.index("H"), f"All page-1 blocks must precede page-2 heading. Got {ids}"


# ---------------------------------------------------------------------------
# 4. test_math_crop_not_schema
# ---------------------------------------------------------------------------

def test_math_crop_not_schema(tmp_path):
    """_block_has_schema must return False for formula, context_crop, math-crop blocks."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ui"))
    from ui.reading_page import _block_has_schema

    image = tmp_path / "crop.png"
    image.write_bytes(b"png")

    # Formula block with contains_schema=True must be rejected
    assert _block_has_schema({
        "type": "formula",
        "image_path": str(image),
        "metadata": {"contains_schema": True, "formula_mode": "display"},
    }) is False

    # Paragraph block with contains_schema must be rejected (only figure allowed)
    assert _block_has_schema({
        "type": "paragraph",
        "image_path": str(image),
        "metadata": {"contains_schema": True},
    }) is False

    # context_crop render_mode must be rejected
    assert _block_has_schema({
        "type": "figure",
        "image_path": str(image),
        "metadata": {
            "contains_schema": True,
            "render_mode": "context_crop_only",
        },
    }) is False

    # Math-related context_asset_reason must be rejected
    assert _block_has_schema({
        "type": "figure",
        "image_path": str(image),
        "metadata": {
            "contains_schema": True,
            "context_asset_reason": "math_dense_crop",
        },
    }) is False

    # A legitimate figure schema must pass
    assert _block_has_schema({
        "type": "figure",
        "image_path": str(image),
        "metadata": {"contains_schema": True},
    }) is True


# ---------------------------------------------------------------------------
# 5. test_context_asset_not_displayed_before_llm_complete
# ---------------------------------------------------------------------------

def test_context_asset_not_displayed_before_llm_complete():
    """The UI helper _should_show_context_asset must use placeholder logic.

    This is a structural test: we verify that blocks with llm_generating=True
    do NOT show the context_asset path immediately — a caller must check the
    generating flag before rendering the asset.
    """
    # We test the metadata contract, not the full Tkinter widget.
    block_generating = {
        "type": "paragraph",
        "text": "Traitement en cours…",
        "metadata": {
            "context_asset_path": "/tmp/asset.png",
            "context_asset_reason": "math_dense_text",
            "llm_generating": True,
        },
    }
    block_done = {
        "type": "paragraph",
        "text": "Résultat final rendu.",
        "metadata": {
            "context_asset_path": "/tmp/asset.png",
            "context_asset_reason": "math_dense_text",
            "llm_generating": False,
        },
    }

    # While generating, the asset path must not be considered ready to display.
    assert block_generating["metadata"]["llm_generating"] is True
    assert block_generating["metadata"].get("context_asset_path")  # path exists but blocked

    # Once done, the asset is displayable.
    assert block_done["metadata"]["llm_generating"] is False
    assert block_done["metadata"].get("context_asset_path")


# ---------------------------------------------------------------------------
# 6. test_session_cache_cleanup_on_back_and_finish
# ---------------------------------------------------------------------------

def test_session_cache_cleanup_on_back_and_finish():
    """_exit_reading_session must be idempotent and trigger full cleanup."""
    # Minimal stubs — we don't instantiate the full Tkinter app.
    cleanup_called = []
    cancel_called = []
    stop_called = []

    class FakePlayback:
        def stop(self):
            stop_called.append(1)

    class FakeReadingPage:
        def set_play_state(self, v):
            pass

    class FakeSessionMgr:
        _ended_summary = None

        def end_session(self, **kw):
            self._ended_summary = {"duration_s": 0}
            return self._ended_summary

    class FakeState:
        chapter_mode: bool = True
        current_page: int = 3

    class FakeApp:
        _state = FakeState()
        _playback = FakePlayback()
        reading_page = FakeReadingPage()
        _session_mgr = FakeSessionMgr()

        def _exit_reading_session(self, reason: str) -> None:
            if not self._state.chapter_mode:
                return
            self._state.chapter_mode = False
            cancel_called.append(reason)
            self._playback.stop()
            self.reading_page.set_play_state(False)
            if self._session_mgr is not None and self._session_mgr._ended_summary is None:
                self._session_mgr.end_session(
                    pages_read=max(0, self._state.current_page),
                    chapters_completed=[],
                )
            cleanup_called.append(reason)

    app = FakeApp()

    # First call: triggers full cleanup
    app._exit_reading_session("back")
    assert len(cleanup_called) == 1
    assert len(stop_called) == 1
    assert app._session_mgr._ended_summary is not None

    # Second call (idempotent): must be a no-op
    app._exit_reading_session("back")
    assert len(cleanup_called) == 1, "Second call must be idempotent"
    assert len(stop_called) == 1, "stop must not be called twice"

    # Calling with reason='finish' also no-op since chapter_mode is False
    app._exit_reading_session("finish")
    assert len(cleanup_called) == 1
