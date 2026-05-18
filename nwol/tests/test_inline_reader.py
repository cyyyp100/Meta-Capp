# tests/test_inline_reader.py
import os
import sys
import time
import tkinter as tk

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.inline_reader import (
    InlineReader,
    _MAX_EMBEDDED_FRAME_WIDTH,
    _block_has_math,
    _context_asset_is_unsafe_for_math_render,
    _formula_crop_display_limits,
    _formula_should_render_with_llm,
    _formula_source_text,
    _markdown_table_to_monospace,
    _should_replace_text_with_context_asset,
    _should_show_context_asset,
)
from ui.reading_page import _block_has_schema
from ui.inline_qa_block import QABlock


def test_markdown_table_to_monospace():
    rendered = _markdown_table_to_monospace("| n | u_n |\n| --- | --- |\n| 1 | 2 |")

    assert "+---+-----+" in rendered
    assert "| n | u_n |" in rendered
    assert "| 1 | 2   |" in rendered


def test_context_asset_display_flag():
    block = {
        "type": "paragraph",
        "text": "Texte avec formule ambiguë",
        "metadata": {
            "context_asset_path": "/tmp/context.png",
            "context_asset_display": True,
            "context_asset_reason": "low_confidence_text",
        },
    }

    assert _should_show_context_asset(block) is True
    block["metadata"]["context_asset_display"] = False
    assert _should_show_context_asset(block) is False


def test_context_asset_hides_math_dense_shows_inline_with_display_flag():
    # math_dense_text is always suppressed (would create too much visual clutter)
    block = {
        "type": "paragraph",
        "text": "Texte mathématique déjà lisible",
        "metadata": {
            "context_asset_path": "/tmp/context.png",
            "context_asset_display": True,
            "context_asset_reason": "math_dense_text",
        },
    }
    assert _should_show_context_asset(block) is False

    # inline_math with context_asset_display=True is now shown (complex inline math crop)
    block["metadata"]["context_asset_reason"] = "inline_math"
    assert _should_show_context_asset(block) is True

    # inline_math without the display flag stays hidden
    block["metadata"]["context_asset_display"] = False
    assert _should_show_context_asset(block) is False

    # low_confidence_text is always shown when asset is present
    block["metadata"]["context_asset_display"] = True
    block["metadata"]["context_asset_reason"] = "low_confidence_text"
    assert _should_show_context_asset(block) is True


def test_wide_inline_context_asset_is_not_used_for_math_render():
    block = {
        "type": "paragraph",
        "text": "$L_{o_i} = L_{bce} + L_{iou}$ framework.",
        "bbox": [50, 667, 545, 714],
        "metadata": {
            "context_asset_path": "/tmp/context.png",
            "context_asset_reason": "inline_math",
            "contains_inline_math": True,
            "formula_mode": "inline",
            "mixed_columns_risk": True,
            "page_width": 612,
        },
    }

    assert _context_asset_is_unsafe_for_math_render(block) is True


def test_context_asset_displays_ambiguous_formula_crop():
    block = {
        "type": "formula",
        "text": "Formule fragmentée",
        "metadata": {
            "context_asset_path": "/tmp/context.png",
            "formula_mode": "ambiguous",
            "context_asset_reason": "inline_math",
        },
    }

    assert _should_show_context_asset(block) is True


def test_context_asset_crop_only_replaces_fragmented_text():
    block = {
        "type": "paragraph",
        "text": "Texte mathématique fragmenté",
        "metadata": {
            "context_asset_path": "/tmp/context.png",
            "context_asset_reason": "fragmented_math_text",
            "render_mode": "context_crop_only",
        },
    }

    assert _should_show_context_asset(block) is True
    assert _should_replace_text_with_context_asset(block) is True


def test_fragmented_context_asset_requires_explicit_display_flag():
    block = {
        "type": "paragraph",
        "text": "Large paragraph with math context kept for the LLM.",
        "metadata": {
            "context_asset_path": "/tmp/context.png",
            "context_asset_reason": "fragmented_math_text",
            "context_asset_display": False,
        },
    }

    assert _should_show_context_asset(block) is False


def test_formula_crop_is_not_treated_as_schema(tmp_path):
    image = tmp_path / "formula.png"
    image.write_bytes(b"png")
    block = {
        "type": "formula",
        "image_path": str(image),
        "metadata": {
            "contains_schema": True,
            "formula_mode": "display",
            "render_mode": "pdf_crop",
        },
    }

    assert _block_has_schema(block) is False


def test_display_formula_crop_prefers_stable_pdf_image_over_latex_llm():
    block = {
        "type": "formula",
        "text": r"\theta_0 = \theta_0 - \beta",
        "image_path": "/tmp/formula.png",
        "metadata": {
            "formula_mode": "display",
            "render_mode": "pdf_crop",
            "wide_initial_crop": True,
        },
    }

    assert _formula_should_render_with_llm(block) is False
    assert _formula_source_text(block).startswith("$$")
    assert _formula_source_text(block).endswith("$$")


def test_formula_crop_display_limits_follow_pdf_bbox_not_png_resolution():
    block = {
        "type": "formula",
        "bbox": [440.0, 603.0, 459.0, 621.0],
        "metadata": {"render_mode": "pdf_crop"},
    }

    max_width, max_height = _formula_crop_display_limits(block)

    assert 88 <= max_width < 120
    assert 42 <= max_height < 80


def test_corrupt_formula_source_uses_image_only_prompt():
    block = {
        "type": "formula",
        "text": r"$\ l a_{bel}{loss L_{o_i} = L_{bce} + L_{iou}}$",
        "metadata": {
            "formula_mode": "display",
            "render_mode": "pdf_crop",
        },
    }

    assert _formula_should_render_with_llm(block) is True
    assert "image jointe" in _formula_source_text(block)
    assert r"\ l a" not in _formula_source_text(block)


def test_block_has_math_uses_metadata_signals():
    assert _block_has_math({
        "type": "paragraph",
        "text": "Suite extraite par OCR sans délimiteurs explicites.",
        "metadata": {"formula_mode": "inline"},
    }) is True
    assert _block_has_math({
        "type": "paragraph",
        "text": "Texte dense détecté en amont sans notation mathématique.",
        "metadata": {"context_asset_reason": "math_dense_text"},
    }) is False
    assert _block_has_math({
        "type": "paragraph",
        "text": "On obtient alors u_n sim n et x^2 tend vers zéro dans ce passage.",
        "metadata": {},
    }) is True
    assert _block_has_math({
        "type": "paragraph",
        "text": (
            "In our experiments, few-shot recognition and expectation-based methods "
            "use Faster / Mask R-CNN [36, 17] without any formula."
        ),
        "metadata": {"context_asset_reason": "math_dense_text"},
    }) is False


def _root_or_skip():
    if os.environ.get("CODEX_CI") == "1":
        pytest.skip("Tk indisponible dans l'environnement Codex headless")
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except tk.TclError as exc:
        pytest.skip(f"Tk indisponible: {exc}")


def _drain_tk(root, iterations: int = 30) -> None:
    for _ in range(iterations):
        root.update()
        time.sleep(0.002)


def _widget_text(widget) -> str:
    chunks = []
    if isinstance(widget, tk.Text):
        chunks.append(widget.get("1.0", "end"))
    elif isinstance(widget, tk.Label):
        chunks.append(widget.cget("text"))
    for child in widget.winfo_children():
        chunks.append(_widget_text(child))
    return "\n".join(chunks)


def test_embed_qa_block_window_create():
    root = _root_or_skip()
    try:
        reader = InlineReader(root)
        reader.pack()
        block = reader.embed_qa_block(
            {
                "question_type": "open",
                "question": "Que signifie ce passage ?",
                "choices": [],
                "expected_answer": "Une explication.",
            },
            on_submit=lambda answer, response_time_ms: None,
            on_rephrase=lambda: None,
        )
        assert block in reader._embedded_frames
        assert len(reader.text.window_names()) == 1
    finally:
        root.destroy()


def test_loading_overlay_can_be_shown_updated_and_hidden():
    root = _root_or_skip()
    try:
        reader = InlineReader(root)
        reader.pack(fill="both", expand=True)

        reader.show_loading_overlay("Préparation du PDF")
        _drain_tk(root)
        overlay = reader._loading_overlay

        assert overlay is not None
        assert overlay.winfo_ismapped()
        assert "Préparation du PDF" in _widget_text(overlay)

        reader.show_loading_overlay("Analyse de la page")
        _drain_tk(root)

        assert reader._loading_overlay is overlay
        assert "Analyse de la page" in _widget_text(overlay)

        reader.hide_loading_overlay()
        _drain_tk(root)

        assert reader._loading_overlay is None
    finally:
        root.destroy()


def test_embed_qa_block_forces_scroll_to_new_question(monkeypatch):
    root = _root_or_skip()
    try:
        reader = InlineReader(root)
        reader.pack()
        calls = []
        monkeypatch.setattr(reader, "scroll_to_bottom", lambda force=False: calls.append(force))

        reader.embed_qa_block(
            {
                "question_type": "open",
                "question": "Que signifie ce passage ?",
                "choices": [],
                "expected_answer": "Une explication.",
            },
            on_submit=lambda answer, response_time_ms: None,
            on_rephrase=lambda: None,
        )

        assert True in calls
    finally:
        root.destroy()


def test_embedded_frame_width_is_capped(monkeypatch):
    root = _root_or_skip()
    try:
        reader = InlineReader(root)
        reader.pack()
        frame = tk.Frame(reader.text)
        monkeypatch.setattr(reader.text, "winfo_width", lambda: 1800)

        reader._resize_frame(frame)

        assert int(frame.cget("width")) == _MAX_EMBEDDED_FRAME_WIDTH
    finally:
        root.destroy()


def test_qa_block_ignores_late_callbacks_after_destroy():
    root = _root_or_skip()
    try:
        block = QABlock(
            root,
            {
                "question_type": "open",
                "question": "Que signifie ce passage ?",
            },
            on_submit=lambda answer, response_time_ms: None,
            on_rephrase=lambda: None,
        )
        block.destroy()

        block.show_loading()
        block.show_new_question({"question_type": "open", "question": "Nouvelle question ?"})
        block.show_feedback("correct", "Exact.")
        block.show_follow_up_answer("Reponse tardive.")
        block.remove_follow_up_form()
    finally:
        root.destroy()


def test_qa_block_keeps_feedback_visible_while_next_question_loads():
    root = _root_or_skip()
    try:
        block = QABlock(
            root,
            {
                "question_type": "open",
                "question": "Que signifie ce passage ?",
            },
            on_submit=lambda answer, response_time_ms: None,
            on_rephrase=lambda: None,
        )
        block.show_loading()
        block.show_pending_question("Génération d'une nouvelle question…")
        block.show_feedback("incorrect", "Non.", hint="Relis le sujet.")
        block.show_pending_question("Génération d'une nouvelle question…")

        text = _widget_text(block)
        assert "Non." in text
        assert "Indice : Relis le sujet." in text
        assert "Génération d'une nouvelle question" in text
    finally:
        root.destroy()


def test_embed_feedback_and_reformulation():
    root = _root_or_skip()
    try:
        reader = InlineReader(root)
        reader.pack()
        reader.embed_feedback("correct", "Exact.", "", "")
        reader.embed_reformulation({
            "rephrasing_angle": "analogie",
            "rephrased_paragraph": "Vu autrement.",
            "note": "Compare les formulations.",
        })
        assert len(reader._embedded_frames) == 2
    finally:
        root.destroy()


def test_bullet_list_renders_math_and_records_current_range(monkeypatch):
    root = _root_or_skip()
    try:
        from core import latex as latex_module

        monkeypatch.setattr(
            latex_module,
            "formula_to_tk_image",
            lambda latex, display=False, max_height=None: tk.PhotoImage(width=1, height=1),
        )

        reader = InlineReader(root)
        reader.pack()
        reader.append_block({
            "type": "bullet_list",
            "items": [
                "La suite est notée $(u_{n})_{n}$.",
                "$u_{n+1}$ est le terme suivant.",
            ],
        })
        root.update()

        assert len(reader._paragraph_ranges) == 1
        assert "$u_{n+1}$ est le terme suivant." in reader._paragraph_ranges[-1]["text"]
        assert "$u_{n+1}$" not in reader.text.get("1.0", "end")
        assert len(reader.text.window_names()) >= 2
    finally:
        root.destroy()


def test_apply_and_reveal_paragraph_mask():
    root = _root_or_skip()
    try:
        reader = InlineReader(root)
        reader.pack()
        for char in "Un concept important\n":
            reader.append_char(char)

        reader.apply_mask(3, 10, "réponse masquée temporairement")
        masked_text = reader.text.get("1.0", "end")

        assert "réponse masquée temporairement" in masked_text
        assert "concept" not in masked_text

        reader.reveal_mask()
        revealed_text = reader.text.get("1.0", "end")

        assert "Un concept important" in revealed_text
    finally:
        root.destroy()


def test_append_char_renders_completed_inline_math_before_newline(monkeypatch):
    root = _root_or_skip()
    try:
        from core import latex as latex_module

        monkeypatch.setattr(
            latex_module,
            "formula_to_tk_image",
            lambda latex, display=False, max_height=None: tk.PhotoImage(width=1, height=1),
        )

        reader = InlineReader(root)
        reader.pack()
        for char in "On a $u_n$":
            reader.append_char(char)
        root.update()

        assert "$u_n$" not in reader.text.get("1.0", "end")
        assert len(reader.text.window_names()) == 1
    finally:
        root.destroy()


def test_completed_char_paragraph_can_be_replaced_by_llm(monkeypatch):
    root = _root_or_skip()
    try:
        from llm import ollama_client

        def fake_render_math_paragraph_stream_async(
            text,
            image_paths,
            on_token,
            on_complete,
            on_error,
            model=ollama_client.OLLAMA_MODEL,
        ):
            root.after(0, lambda: on_token("On a "))
            root.after(0, lambda: on_token("$u_n \\sim n$."))
            root.after(0, lambda: on_complete("On a $u_n \\sim n$."))
            return None

        monkeypatch.setattr(
            ollama_client,
            "render_math_paragraph_stream_async",
            fake_render_math_paragraph_stream_async,
        )

        reader = InlineReader(root)
        reader.set_llm_speed(1)
        reader.pack()
        for char in "On a u_n sim n.\n":
            reader.append_char(char)

        started = reader.render_completed_paragraph_with_llm({
            "type": "paragraph",
            "text": "On a u_n sim n.",
            "metadata": {"formula_mode": "inline"},
        })
        _drain_tk(root)

        assert started is True
        rendered_text = reader.text.get("1.0", "end")
        assert "Rendu mathématique" not in rendered_text
        assert "On a " in rendered_text
    finally:
        root.destroy()


def test_math_paragraph_keeps_initial_render_while_llm_is_pending(monkeypatch):
    root = _root_or_skip()
    callbacks = {}
    try:
        from llm import ollama_client

        def fake_render_math_paragraph_stream_async(
            text,
            image_paths,
            on_token,
            on_complete,
            on_error,
            model=ollama_client.OLLAMA_MODEL,
        ):
            callbacks["token"] = on_token
            callbacks["complete"] = on_complete
            return None

        monkeypatch.setattr(
            ollama_client,
            "render_math_paragraph_stream_async",
            fake_render_math_paragraph_stream_async,
        )

        reader = InlineReader(root)
        reader.set_llm_speed(1)
        reader.pack()
        reader.append_block({
            "type": "paragraph",
            "text": "On a u_n sim n.",
            "metadata": {"formula_mode": "inline"},
        })
        root.update()

        pending_text = reader.text.get("1.0", "end")
        assert "On a u_n sim n." in pending_text
        assert "Rendu mathématique" not in pending_text

        callbacks["token"]("On a ")
        callbacks["token"]("$u_n \\sim n$.")
        callbacks["complete"]("On a $u_n \\sim n$.")
        _drain_tk(root)

        rendered_text = reader.text.get("1.0", "end")
        assert "Rendu mathématique" not in rendered_text
        assert "On a " in rendered_text
        assert "sim n" not in rendered_text
    finally:
        root.destroy()


def test_stream_math_paragraph_inserts_tokens_and_records_final_text(monkeypatch):
    root = _root_or_skip()
    completed = []
    try:
        from llm import ollama_client

        def fake_render_math_paragraph_stream_async(
            text,
            image_paths,
            on_token,
            on_complete,
            on_error,
            model=ollama_client.OLLAMA_MODEL,
        ):
            root.after(0, lambda: on_token("On a "))
            root.after(0, lambda: on_token("$u_n \\sim n$."))
            root.after(0, lambda: on_complete("On a $u_n \\sim n$."))
            return None

        monkeypatch.setattr(
            ollama_client,
            "render_math_paragraph_stream_async",
            fake_render_math_paragraph_stream_async,
        )

        reader = InlineReader(root)
        reader.set_llm_speed(1)
        reader.pack()
        reader.stream_math_paragraph(
            {
                "type": "paragraph",
                "text": "On a u_n sim n.",
                "metadata": {"formula_mode": "inline"},
            },
            on_complete=lambda text: completed.append(text),
        )
        _drain_tk(root)

        rendered_text = reader.text.get("1.0", "end")
        assert completed == ["On a $u_n \\sim n$."]
        assert "Traitement en cours" not in rendered_text
        assert "On a " in rendered_text
        assert reader._paragraph_ranges[-1]["text"] == "On a $u_n \\sim n$."
    finally:
        root.destroy()


def test_stream_math_paragraph_types_final_response_when_no_tokens(monkeypatch):
    root = _root_or_skip()
    completed = []
    try:
        from llm import ollama_client

        def fake_render_math_paragraph_stream_async(
            text,
            image_paths,
            on_token,
            on_complete,
            on_error,
            model=ollama_client.OLLAMA_MODEL,
        ):
            root.after(0, lambda: on_complete("ABC"))
            return None

        monkeypatch.setattr(
            ollama_client,
            "render_math_paragraph_stream_async",
            fake_render_math_paragraph_stream_async,
        )

        reader = InlineReader(root)
        reader.set_llm_speed(20)
        reader.pack()
        reader.stream_math_paragraph(
            {
                "type": "paragraph",
                "text": "On a u_n sim n.",
                "metadata": {"formula_mode": "inline"},
            },
            on_complete=lambda text: completed.append(text),
        )
        root.update()

        initial_text = reader.text.get("1.0", "end")
        assert "ABC" not in initial_text

        _drain_tk(root, iterations=140)

        rendered_text = reader.text.get("1.0", "end")
        assert "ABC" in rendered_text
        assert completed == ["ABC"]
    finally:
        root.destroy()


def test_stream_image_description_keeps_following_reader_content():
    root = _root_or_skip()
    callbacks = {}
    try:
        def fake_render_async(image_path, caption, on_token, on_complete, on_error):
            callbacks["complete"] = on_complete

        reader = InlineReader(root)
        reader.set_llm_speed(1)
        reader.pack()
        reader._stream_image_description(
            block={"type": "figure"},
            image_path="/tmp/schema.png",
            caption="",
            loading_text="Analyse du schéma…",
            loading_tag="schema_loading",
            final_tag="schema_description",
            render_async=fake_render_async,
        )
        root.update()

        reader.append_block({"type": "paragraph", "text": "Contenu suivant conservé."})
        root.update()

        callbacks["complete"]("Description du schéma.")
        _drain_tk(root, iterations=80)

        rendered_text = reader.text.get("1.0", "end")
        assert "Description du schéma." in rendered_text
        assert "Contenu suivant conservé." in rendered_text
        assert "Analyse du schéma" not in rendered_text
    finally:
        root.destroy()


def test_stream_math_paragraph_renders_formula_before_completion(monkeypatch):
    root = _root_or_skip()
    completed = []
    callbacks = {}
    try:
        from core import latex as latex_module
        from llm import ollama_client

        monkeypatch.setattr(
            latex_module,
            "formula_to_tk_image",
            lambda latex, display=False, max_height=None: tk.PhotoImage(width=1, height=1),
        )

        def fake_render_math_paragraph_stream_async(
            text,
            image_paths,
            on_token,
            on_complete,
            on_error,
            model=ollama_client.OLLAMA_MODEL,
        ):
            callbacks["complete"] = on_complete
            root.after(0, lambda: on_token("On a $u_n$"))
            return None

        monkeypatch.setattr(
            ollama_client,
            "render_math_paragraph_stream_async",
            fake_render_math_paragraph_stream_async,
        )

        reader = InlineReader(root)
        reader.set_llm_speed(1)
        reader.pack()
        reader.stream_math_paragraph(
            {
                "type": "paragraph",
                "text": "On a u_n.",
                "metadata": {"formula_mode": "inline"},
            },
            on_complete=lambda text: completed.append(text),
        )
        _drain_tk(root, iterations=80)

        assert completed == []
        assert "$u_n$" not in reader.text.get("1.0", "end")
        assert len(reader.text.window_names()) == 1

        callbacks["complete"]("On a $u_n$")
        _drain_tk(root, iterations=20)
        assert completed == ["On a $u_n$"]
    finally:
        root.destroy()
