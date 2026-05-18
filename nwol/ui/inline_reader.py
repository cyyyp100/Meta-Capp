# ui/inline_reader.py — Lecture progressive avec Q&R inline
from __future__ import annotations

import logging
import re
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

from collections import deque

from config.settings import (
    ASSETS_DIR,
    LLM_CHAR_SPEED_MS,
    UI_FONT_FAMILY,
    UI_FONT_SIZE,
    UI_MONO_FONT,
)
from document.postprocess.latex_quality import latex_looks_corrupt
from i18n import t
from ui import theme
from ui.components import LoadingState
from ui.inline_qa_block import QABlock
from ui.rich_text import MATH_PATTERN, render_rich_text as _render_rich_text, rich_text_widget as _rich_text_widget

logger = logging.getLogger("UI.inline_reader")

_MAX_EMBEDDED_FRAME_WIDTH = 1120
_MAX_READER_MEDIA_WIDTH = 760
_MAX_READER_CONTEXT_HEIGHT = 190
_MAX_READER_FIGURE_HEIGHT = 220
_MAX_READER_FORMULA_WIDTH = 660
_MAX_READER_FORMULA_HEIGHT = 92
_MAX_READER_FORMULA_CROP_HEIGHT = 200
_MAX_READER_INLINE_FORMULA_HEIGHT = 42
_MAX_READER_TABLE_HEIGHT = 260
_MAX_READER_FIGURE_CROP_HEIGHT = 400

_LATEX_SIGNAL_RE = re.compile(r"\\[A-Za-z]{2,}|\$[^$\n]{1,200}\$")
_MATH_TEXT_SIGNAL_RE = re.compile(
    r"(?<!\w)[A-Za-z]\s*[_^]\s*[A-Za-z0-9{(]"
    r"|[∑∫√∞≈≠≤≥→←↔∈∉∀∃αβγδλμσφψω]"
    r"|\b(?:lim|sin|cos|tan|ln|log|exp)\b\s*[_({]?"
    r"|[A-Za-z0-9]\s*(?:=|≤|≥|≈|≠|<|>)\s*[A-Za-z0-9\\∑∫√∞αβγδλμσφψω]",
)


def _block_has_math(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if metadata.get("contains_inline_math"):
        return True
    if metadata.get("formula_mode") in {"inline", "ambiguous"}:
        return True
    if metadata.get("context_asset_reason") == "inline_math":
        return True
    text = block.get("text") or ""
    if len(text) < 20:
        return False
    return bool(
        MATH_PATTERN.search(text)
        or _LATEX_SIGNAL_RE.search(text)
        or _MATH_TEXT_SIGNAL_RE.search(text)
    )


def _new_incremental_math_state(prefix: str) -> dict:
    return {
        "prefix": prefix,
        "counter": 0,
        "pending_mark": None,
        "start_mark": None,
        "mode": None,
        "raw": "",
        "rendered_any": False,
    }


def _insert_incremental_math_char(
    widget: tk.Text,
    char: str,
    insert_pos: str,
    text_tag: str,
    state: dict,
) -> None:
    insert_mark = _next_incremental_mark(state, "insert")
    widget.mark_set(insert_mark, insert_pos)
    widget.mark_gravity(insert_mark, tk.LEFT)
    widget.insert(insert_pos, char, text_tag)
    before = widget.index(insert_mark)
    after = widget.index(f"{before}+1c")
    widget.mark_unset(insert_mark)

    pending_mark = state.get("pending_mark")
    if pending_mark:
        if char == "$":
            state["start_mark"] = pending_mark
            state["mode"] = "$$"
            state["raw"] = "$$"
            state["pending_mark"] = None
            return
        state["start_mark"] = pending_mark
        state["mode"] = "$"
        state["raw"] = "$" + char
        state["pending_mark"] = None
        return

    mode = state.get("mode")
    if mode is None:
        if char == "$":
            mark = _next_incremental_mark(state, "math_start")
            widget.mark_set(mark, before)
            widget.mark_gravity(mark, tk.LEFT)
            state["pending_mark"] = mark
        return

    state["raw"] = str(state.get("raw") or "") + char
    raw = state["raw"]
    should_close = (
        mode == "$"
        and len(raw) > 1
        and raw.endswith("$")
        and not _dollar_is_escaped(raw, len(raw) - 1)
    ) or (
        mode == "$$"
        and len(raw) > 3
        and raw.endswith("$$")
        and not _dollar_is_escaped(raw, len(raw) - 2)
    )
    if should_close:
        _replace_incremental_math_span(widget, raw, after, text_tag, state)


def _replace_incremental_math_span(widget: tk.Text, raw: str, end_index: str, text_tag: str, state: dict) -> None:
    start_mark = state.get("start_mark")
    if not start_mark:
        _reset_incremental_math_span(state)
        return
    try:
        start = widget.index(start_mark)
    except tk.TclError:
        _reset_incremental_math_span(state)
        return

    widget.delete(start, end_index)
    _render_rich_text(widget, raw, insert_pos=start, text_tag=text_tag)
    state["rendered_any"] = True
    try:
        widget.mark_unset(start_mark)
    except tk.TclError:
        pass
    _reset_incremental_math_span(state)


def _reset_incremental_math_span(state: dict) -> None:
    state["pending_mark"] = None
    state["start_mark"] = None
    state["mode"] = None
    state["raw"] = ""


def _next_incremental_mark(state: dict, kind: str) -> str:
    state["counter"] = int(state.get("counter") or 0) + 1
    return f"_{state.get('prefix', 'math')}_{kind}_{state['counter']}"


def _dollar_is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


class InlineReader(tk.Frame):
    def __init__(self, master, on_paragraph_rephrase=None, **kwargs):
        super().__init__(
            master,
            bg=theme.SURFACE,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            **kwargs,
        )
        self._image_refs: list = []
        self._rich_window_refs: list = []
        self._embedded_frames: list[tk.Frame] = []
        self._pdf_path: str | None = None
        self.paragraph_mask: dict | None = None
        self._paragraph_ranges: list[dict] = []
        self._active_paragraph_start: str | None = None
        self._active_paragraph_chars: list[str] = []
        self._active_math_state = _new_incremental_math_state("char_paragraph")
        self._render_generation: int = 0
        self._llm_char_speed_ms: int = LLM_CHAR_SPEED_MS
        self._on_paragraph_rephrase = on_paragraph_rephrase
        self._auto_follow_bottom = True
        self._loading_overlay: LoadingState | None = None
        self._build_widget()

    def _build_widget(self) -> None:
        self.text = tk.Text(
            self,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg=theme.SURFACE,
            fg=theme.TEXT,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE),
            padx=44,
            pady=32,
            spacing1=5,
            spacing3=10,
            relief=tk.FLAT,
            borderwidth=0,
            cursor="arrow",
            insertbackground=theme.TEXT,
        )
        self.text._image_refs = self._image_refs
        self.text._window_refs = self._rich_window_refs
        scrollbar = tk.Scrollbar(self, command=self._on_scrollbar_command)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text.bind("<Configure>", self._resize_embedded_frames)
        self.text.bind("<MouseWheel>", self._on_user_scroll, add="+")
        self.text.bind("<Button-4>", self._on_user_scroll, add="+")
        self.text.bind("<Button-5>", self._on_user_scroll, add="+")
        self._configure_tags()

    def _configure_tags(self) -> None:
        base_font = tkfont.Font(family=UI_FONT_FAMILY, size=UI_FONT_SIZE)
        h1_font = tkfont.Font(family=UI_FONT_FAMILY, size=UI_FONT_SIZE + 6, weight="bold")
        h2_font = tkfont.Font(family=UI_FONT_FAMILY, size=UI_FONT_SIZE + 3, weight="bold")
        h3_font = tkfont.Font(family=UI_FONT_FAMILY, size=UI_FONT_SIZE + 1, weight="bold")
        mono_font = tkfont.Font(family=UI_MONO_FONT, size=UI_FONT_SIZE - 1)

        self.text.tag_configure("h1", font=h1_font, foreground=theme.ACCENT_HOVER, justify="center", spacing1=18, spacing3=12)
        self.text.tag_configure("h2", font=h2_font, foreground=theme.ACCENT_HOVER, spacing1=14, spacing3=8)
        self.text.tag_configure("h3", font=h3_font, foreground=theme.TEXT_SOFT, spacing1=12, spacing3=6)
        self.text.tag_configure("paragraph", font=base_font, lmargin1=24, lmargin2=24, rmargin=20, spacing3=10)
        self.text.tag_configure("bullet_list", font=base_font, lmargin1=42, lmargin2=58, rmargin=20, spacing3=10)
        self.text.tag_configure("code", font=mono_font, background="#F2F6F8", lmargin1=24, lmargin2=24, rmargin=20, spacing1=6, spacing3=8)
        self.text.tag_configure("table", font=mono_font, background="#F7FAFC", lmargin1=24, lmargin2=24, rmargin=20, spacing1=6, spacing3=10)
        self.text.tag_configure("caption", font=base_font, foreground=theme.MUTED, justify="center", spacing1=3, spacing3=10)
        self.text.tag_configure("media", justify="center", spacing1=8, spacing3=12)
        self.text.tag_configure("note", font=(theme.FONT_UI, 10, "italic"), foreground=theme.MUTED, lmargin1=24, lmargin2=24)
        self.text.tag_configure("masked", background="#E8EDF0", foreground=theme.MUTED, font=(UI_FONT_FAMILY, UI_FONT_SIZE, "italic"))
        self.text.tag_configure("callout_label", font=(UI_FONT_FAMILY, UI_FONT_SIZE - 1, "bold"), foreground=theme.ACCENT_HOVER, lmargin1=32, spacing1=10, spacing3=2)
        self.text.tag_configure("callout_body", font=base_font, lmargin1=32, lmargin2=32, rmargin=20, spacing3=10)
        self.text.tag_configure(
            "math_loading",
            foreground=theme.MUTED,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE, "italic"),
            lmargin1=24,
            lmargin2=24,
        )
        self.text.tag_configure(
            "schema_loading",
            foreground=theme.MUTED,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE - 1, "italic"),
            lmargin1=24,
            lmargin2=24,
        )
        self.text.tag_configure(
            "schema_description",
            foreground=theme.TEXT_SOFT,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE - 1),
            lmargin1=24,
            lmargin2=24,
            rmargin=20,
            spacing3=8,
        )
        self.text.tag_configure(
            "table_loading",
            foreground=theme.MUTED,
            font=(UI_FONT_FAMILY, UI_FONT_SIZE - 1, "italic"),
            lmargin1=24,
            lmargin2=24,
        )
        self.text.tag_configure(
            "table_description",
            foreground=theme.TEXT_SOFT,
            font=mono_font,
            lmargin1=24,
            lmargin2=24,
            rmargin=20,
            spacing1=6,
            spacing3=10,
        )

    def clear(self, *, hide_loading: bool = True) -> None:
        if hide_loading:
            self.hide_loading_overlay()
        self._image_refs.clear()
        self._rich_window_refs.clear()
        self._embedded_frames.clear()
        self.paragraph_mask = None
        self._paragraph_ranges.clear()
        self._render_generation += 1
        self._auto_follow_bottom = True
        self._active_paragraph_start = None
        self._active_paragraph_chars.clear()
        self._active_math_state = _new_incremental_math_state("char_paragraph")
        self._write(lambda: self.text.delete("1.0", tk.END))

    def set_pdf_path(self, pdf_path: str | None) -> None:
        self._pdf_path = pdf_path

    def set_llm_speed(self, ms: int) -> None:
        self._llm_char_speed_ms = max(1, int(ms))

    def show_loading_overlay(self, text: str = "Préparation du PDF") -> None:
        if self._loading_overlay is not None:
            try:
                if not self._loading_overlay.winfo_exists():
                    self._loading_overlay = None
            except tk.TclError:
                self._loading_overlay = None

        if self._loading_overlay is None:
            overlay = LoadingState(
                self.text,
                text=text,
                bg=theme.SURFACE_SOFT,
                highlightthickness=1,
                highlightbackground=theme.BORDER_STRONG,
                highlightcolor=theme.BORDER_STRONG,
            )
            self._loading_overlay = overlay

        self._loading_overlay.place(relx=0.5, rely=0.5, anchor="center")
        self._loading_overlay.tkraise()
        self._loading_overlay.start(text)

    def hide_loading_overlay(self) -> None:
        overlay = self._loading_overlay
        self._loading_overlay = None
        if overlay is None:
            return
        try:
            if overlay.winfo_exists():
                overlay.destroy()
        except tk.TclError:
            pass

    def append_char(self, char: str) -> None:
        self.hide_loading_overlay()

        def _insert() -> None:
            if char == "\n":
                if self._active_paragraph_start is not None:
                    paragraph_text = "".join(self._active_paragraph_chars)
                    end_index = self.text.index("end-1c")
                    if MATH_PATTERN.search(paragraph_text) and not self._active_math_state["rendered_any"]:
                        self.text.delete(self._active_paragraph_start, end_index)
                        _render_rich_text(
                            self.text,
                            paragraph_text,
                            insert_pos=self._active_paragraph_start,
                            text_tag="paragraph",
                        )
                        end_index = self.text.index("end-1c")
                    self._paragraph_ranges.append({
                        "start": self._active_paragraph_start,
                        "end": end_index,
                        "text": paragraph_text,
                    })
                    self._active_paragraph_start = None
                    self._active_paragraph_chars.clear()
                    self._active_math_state = _new_incremental_math_state("char_paragraph")
                self.text.insert(tk.END, char, "paragraph")
                return

            if self._active_paragraph_start is None:
                self._active_paragraph_start = self.text.index("end-1c")
                self._active_paragraph_chars.clear()
                self._active_math_state = _new_incremental_math_state("char_paragraph")
            self._active_paragraph_chars.append(char)
            _insert_incremental_math_char(
                self.text,
                char,
                insert_pos=tk.END,
                text_tag="paragraph",
                state=self._active_math_state,
            )

        self._write(_insert)
        self.scroll_to_bottom()

    def append_block(self, block: dict) -> None:
        self.hide_loading_overlay()
        btype = block.get("type", "")
        if btype in {"heading", "subheading", "subsubheading"}:
            default_level = {"heading": 1, "subheading": 2, "subsubheading": 3}.get(btype, 1)
            raw_level = block.get("level")
            if raw_level is None:
                level = default_level
            else:
                try:
                    level = min(3, max(1, int(raw_level)))
                except (TypeError, ValueError):
                    level = default_level
            tag = f"h{level}"
            self._write(lambda t=block.get("text", ""), g=tag: self.text.insert(tk.END, "\n" + t + "\n", g))
        elif btype == "formula":
            self._insert_formula(block)
        elif btype == "figure":
            self._insert_figure(block)
        elif btype == "code":
            text = block.get("text", "")
            self._write(lambda t=text: self.text.insert(tk.END, "\n" + t + "\n", "code"))
        elif btype == "table":
            self._insert_table(block)
        elif btype == "bullet_list":
            self._insert_bullet_list(block)
        elif btype in {"definition", "theorem", "example", "remark", "warning", "exercise", "question"}:
            self._insert_callout(block)
        elif btype in {"paragraph", "text", "quote", "abstract"}:
            self._insert_paragraph(block)
        else:
            self._insert_paragraph(block)
        self.scroll_to_bottom()

    def embed_qa_block(self, question: dict, on_submit, on_rephrase, on_reveal_mask=None) -> QABlock:
        block = QABlock(
            self.text,
            question,
            on_submit=on_submit,
            on_rephrase=on_rephrase,
            on_reveal_mask=on_reveal_mask,
        )
        self._insert_embedded_frame(block)
        self.scroll_to_bottom(force=True)
        self.after_idle(lambda: self.scroll_to_bottom(force=True))
        return block

    def embed_paragraph_rephrase_button(self, block: dict) -> None:
        """Insère le bouton ↻ Reformuler après le paragraphe (avant le QABlock).

        Appelé depuis reading_page._continue_after_paragraph_render pour couvrir
        aussi bien le mode char que le mode block.
        """
        if self._on_paragraph_rephrase is None:
            return
        widget = _ParagraphRephraseWidget(self.text, block, self._on_paragraph_rephrase)
        self._insert_embedded_frame(widget)

    def apply_mask(self, start_char: int, end_char: int, placeholder: str) -> None:
        if not self._paragraph_ranges:
            return
        self.reveal_mask()
        paragraph = self._paragraph_ranges[-1]
        paragraph_text = paragraph.get("text", "")
        start_char = max(0, int(start_char))
        end_char = min(len(paragraph_text), int(end_char))
        if end_char <= start_char:
            return

        start_index = f"{paragraph['start']}+{start_char}c"
        end_index = f"{paragraph['start']}+{end_char}c"
        original_text = self.text.get(start_index, end_index)
        placeholder = (placeholder or t("qa.mask_placeholder")).strip()
        if not original_text or not placeholder:
            return

        start_mark = "_paragraph_mask_start"
        end_mark = "_paragraph_mask_end"

        def _apply() -> None:
            self.text.delete(start_index, end_index)
            self.text.insert(start_index, placeholder, ("paragraph", "masked"))
            self.text.mark_set(start_mark, start_index)
            self.text.mark_gravity(start_mark, tk.LEFT)
            self.text.mark_set(end_mark, f"{start_mark}+{len(placeholder)}c")
            self.text.mark_gravity(end_mark, tk.RIGHT)
            self.text.tag_add("masked", start_mark, end_mark)

        self._write(_apply)
        self.paragraph_mask = {
            "start_mark": start_mark,
            "end_mark": end_mark,
            "original_text": original_text,
        }

    def reveal_mask(self) -> None:
        if not self.paragraph_mask:
            return
        mask = self.paragraph_mask
        self.paragraph_mask = None
        start_mark = mask.get("start_mark")
        end_mark = mask.get("end_mark")
        original_text = mask.get("original_text", "")
        if not start_mark or not end_mark:
            return

        def _reveal() -> None:
            try:
                start_index = self.text.index(start_mark)
                end_index = self.text.index(end_mark)
            except tk.TclError:
                return
            self.text.delete(start_index, end_index)
            self.text.insert(start_index, original_text, "paragraph")
            for mark in (start_mark, end_mark):
                try:
                    self.text.mark_unset(mark)
                except tk.TclError:
                    pass

        self._write(_reveal)

    def embed_feedback(self, verdict: str, feedback: str, completion: str = "", hint: str = "") -> tk.Frame:
        bg = theme.SUCCESS_SOFT if verdict in {"correct", "partial"} else theme.DANGER_SOFT
        border = theme.SUCCESS if verdict in {"correct", "partial"} else theme.DANGER
        frame = _message_frame(self.text, bg, border)
        content = frame._content
        symbol = "✓" if verdict in {"correct", "partial"} else "✗"
        text = f"{symbol} {feedback}"
        if completion:
            text += f"\n{completion}"
        if verdict == "incorrect" and hint:
            text += "\n" + t("qa.hint_prefix", text=hint)
        _rich_text_widget(
            content,
            text,
            bg=bg,
            fg=theme.TEXT,
            font=(theme.FONT_UI, 10),
        ).pack(fill="x", padx=14, pady=12)
        self._insert_embedded_frame(frame)
        return frame

    def embed_reformulation(self, rephrasing: dict) -> tk.Frame:
        frame = _message_frame(self.text, theme.WARNING_SOFT, theme.WARNING)
        body = frame._content
        tk.Label(
            body,
            text=f"🔄 Vu autrement : {rephrasing.get('rephrasing_angle', '')}",
            bg=theme.WARNING_SOFT,
            fg=theme.WARNING,
            font=(theme.FONT_UI, 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 0))
        _rich_text_widget(
            body,
            _normalize_llm_text(rephrasing.get("rephrased_paragraph", "")),
            bg=theme.WARNING_SOFT,
            fg=theme.TEXT,
            font=(theme.FONT_UI, 10),
            justify="left",
        ).pack(fill="x", padx=14, pady=(6, 0))
        note = rephrasing.get("note")
        if note:
            _rich_text_widget(
                body,
                f"Note : {_normalize_llm_text(note)}",
                bg=theme.WARNING_SOFT,
                fg=theme.WARNING,
                font=(theme.FONT_UI, 9, "italic"),
                justify="left",
            ).pack(fill="x", padx=14, pady=(6, 12))

        self._insert_embedded_frame(frame)
        return frame

    def embed_flashcard_notif(self, card: dict | None = None) -> tk.Frame:
        frame = _message_frame(self.text, theme.SUCCESS_SOFT, theme.SUCCESS)
        content = frame._content
        tk.Label(
            content,
            text="📚 Flash card créée",
            bg=theme.SUCCESS_SOFT,
            fg=theme.SUCCESS,
            font=(theme.FONT_UI, 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=10)
        self._insert_embedded_frame(frame)
        return frame

    def embed_status(self, text: str) -> tk.Frame:
        frame = _message_frame(self.text, theme.SURFACE_SOFT, theme.BORDER_STRONG)
        content = frame._content
        _rich_text_widget(
            content,
            text,
            bg=theme.SURFACE_SOFT,
            fg=theme.MUTED,
            font=(theme.FONT_UI, 10, "italic"),
            justify="left",
        ).pack(fill="x", padx=14, pady=10)
        self._insert_embedded_frame(frame)
        return frame

    def embed_chapter_summary(self, summary: dict) -> tk.Frame:
        chapter_summary = summary.get("chapter_summary") or summary
        frame = _message_frame(self.text, theme.ACCENT_SOFT, theme.ACCENT)
        content = frame._content

        title = chapter_summary.get("title") or "Fin de chapitre"
        tk.Label(
            content,
            text=f"◆ Synthèse — {title}",
            bg=theme.ACCENT_SOFT,
            fg=theme.ACCENT_HOVER,
            font=(theme.FONT_UI, 12, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))

        overview = chapter_summary.get("overview", "")
        if overview:
            _rich_text_widget(
                content,
                overview,
                bg=theme.ACCENT_SOFT,
                fg=theme.TEXT,
                font=(theme.FONT_UI, 10, "italic"),
                justify="left",
            ).pack(fill="x", padx=14, pady=(0, 8))

        for index, item in enumerate(chapter_summary.get("recap_qa") or [], start=1):
            question = item.get("question", "")
            answer = item.get("answer", "")
            _rich_text_widget(
                content,
                f"{index}. {question}\n{answer}",
                bg=theme.ACCENT_SOFT,
                fg=theme.TEXT,
                font=(theme.FONT_UI, 10),
                justify="left",
            ).pack(fill="x", padx=14, pady=(0, 8))

        self._insert_embedded_frame(frame)
        return frame

    def scroll_to_bottom(self, force: bool = False) -> None:
        if not (force or self._auto_follow_bottom or self._is_at_bottom()):
            return
        self.text.see(tk.END)
        self.text.yview_moveto(1.0)
        self._auto_follow_bottom = True

    def _on_scrollbar_command(self, *args) -> None:
        self._auto_follow_bottom = False
        self.text.yview(*args)
        self.after_idle(self._refresh_auto_follow_state)

    def _on_user_scroll(self, _event=None) -> None:
        self._auto_follow_bottom = False
        self.after_idle(self._refresh_auto_follow_state)

    def _refresh_auto_follow_state(self) -> None:
        self._auto_follow_bottom = self._is_at_bottom()

    def _is_at_bottom(self) -> bool:
        try:
            _first, last = self.text.yview()
        except tk.TclError:
            return True
        return last >= 0.995

    def _insert_callout(self, block: dict) -> None:
        btype = block.get("type", "remark")
        labels = {
            "definition": "Définition",
            "theorem": "Théorème",
            "example": "Exemple",
            "remark": "Remarque",
            "warning": "Attention",
            "exercise": "Exercice",
            "question": "Question",
        }
        colors = {
            "definition": (theme.ACCENT_SOFT, theme.ACCENT_HOVER),
            "theorem": (theme.ACCENT_SOFT, theme.ACCENT_HOVER),
            "example": (theme.SUCCESS_SOFT, theme.SUCCESS),
            "remark": (theme.SURFACE_SOFT, theme.BORDER_STRONG),
            "warning": (theme.WARNING_SOFT, theme.WARNING),
            "exercise": (theme.ACCENT_SOFT, theme.ACCENT),
            "question": (theme.QUESTION, theme.QUESTION_BORDER),
        }
        label = labels.get(btype, btype.capitalize())
        bg, fg = colors.get(btype, (theme.SURFACE_SOFT, theme.BORDER_STRONG))
        text = (block.get("text") or "").strip()
        if not text:
            return

        frame = _callout_frame(self.text, label, text, bg, fg)
        self._insert_embedded_frame(frame)

    def _insert_paragraph(self, block: dict) -> None:
        if _should_replace_text_with_context_asset(block):
            text = (block.get("text") or "").strip()
            context_path = (block.get("metadata") or {}).get("context_asset_path")
            if text and context_path:
                # Always route broken blocks through LLM + PDF image crop,
                # regardless of text length. Suppresses raw crop to avoid doublon.
                meta = block.setdefault("metadata", {})
                if isinstance(meta, dict):
                    meta["context_asset_display"] = False
                self._insert_math_paragraph_with_llm(block, text)
                return
            if self.embed_context_asset(block):
                return
            if not text:
                return
            # Fall through: no crop — render raw text as best-effort

        text = (block.get("text") or "").strip()
        if not text:
            return

        if _block_has_math(block):
            self._insert_math_paragraph_with_llm(block, text)
            return

        def _insert(t=text):
            self.text.insert(tk.END, t, "paragraph")
            self.text.insert(tk.END, "\n", "paragraph")

        self._write(_insert)
        self.embed_context_asset(block)

    def _insert_math_paragraph_with_llm(self, block: dict, text: str) -> None:
        self._start_math_paragraph_render(block, text, replace_range=None)

    def stream_math_paragraph(
        self,
        block: dict,
        on_token=None,
        on_complete=None,
        use_llm: bool = True,
    ) -> None:
        text = (block.get("text") or "").strip()
        metadata = block.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            block["metadata"] = metadata
        metadata["reader_math_streamed"] = True
        if text:
            metadata.setdefault("reader_source_text", text)
            metadata.setdefault("qa_source_text", text)

        if not text:
            if on_complete:
                self.after(0, lambda: on_complete(None))
            return

        if not use_llm:
            self._insert_streamed_math_fallback(block, text, on_complete)
            return

        self._start_math_paragraph_render(
            block,
            text,
            replace_range=None,
            on_token=on_token,
            on_complete=on_complete,
            placeholder_text=t("reading.processing"),
            placeholder_tag="math_loading",
            insert_initial_text=False,
            append_newline_on_complete=True,
        )

    def stream_schema_description(self, block: dict, use_llm: bool = True, on_done=None) -> None:
        if not use_llm:
            if on_done is not None:
                self.after(0, on_done)
            return
        image_path = _block_render_image_path(block)
        if not image_path:
            if on_done is not None:
                self.after(0, on_done)
            return
        resolved = self._resolve_image_path(image_path)
        if resolved is not None:
            image_path = str(resolved)

        from llm.ollama_client import render_schema_stream_async

        caption = str(block.get("caption") or block.get("text") or "")
        self._stream_image_description(
            block=block,
            image_path=image_path,
            caption=caption,
            loading_text=t("reading.schema_loading"),
            loading_tag="schema_loading",
            final_tag="schema_description",
            render_async=render_schema_stream_async,
            on_done=on_done,
        )

    def _stream_table_description(self, block: dict) -> None:
        image_path = _block_render_image_path(block, preferred_metadata_key="table_image_path")
        if not image_path:
            return
        resolved = self._resolve_image_path(image_path)
        if resolved is not None:
            image_path = str(resolved)

        from llm.ollama_client import render_table_stream_async

        metadata = block.get("metadata") or {}
        caption = str(block.get("caption") or metadata.get("caption") or metadata.get("title") or "")
        self._stream_image_description(
            block=block,
            image_path=image_path,
            caption=caption,
            loading_text=t("reading.table_loading"),
            loading_tag="table_loading",
            final_tag="table_description",
            render_async=render_table_stream_async,
        )

    def _stream_image_description(
        self,
        block: dict,
        image_path: str,
        caption: str,
        loading_text: str,
        loading_tag: str,
        final_tag: str,
        render_async,
        on_done=None,
    ) -> None:
        idx = len(self._paragraph_ranges)
        mark_start = f"_{final_tag}_s_{id(block)}_{idx}"
        mark_end = f"_{final_tag}_e_{id(block)}_{idx}"
        generation = self._render_generation
        state = {"started": False, "finished": False}

        def _insert_placeholder() -> None:
            pos = self.text.index("end-1c")
            self.text.mark_set(mark_start, pos)
            self.text.mark_gravity(mark_start, tk.LEFT)
            self.text.insert(tk.END, "\n" + loading_text + "\n", loading_tag)
            self.text.mark_set(mark_end, "end-1c")
            self.text.mark_gravity(mark_end, tk.LEFT)
            # Keep later reader content outside the async replacement range.
            # Without this spacer, content appended at Tk's end index can share
            # mark_end and be deleted when the LLM description finalizes.
            self.text.insert(tk.END, "\n", loading_tag)

        self._write(_insert_placeholder)
        self.scroll_to_bottom()

        char_q: deque[str] = deque()
        pump_active = [False]
        llm_done = [False]
        display_ref: list[str] = [""]
        streamed_any = [False]

        def _do_replace() -> None:
            display = display_ref[0]

            def _replace() -> None:
                try:
                    start = self.text.index(mark_start)
                    end = self.text.index(mark_end)
                except tk.TclError:
                    return
                self.text.delete(start, end)
                self.text.mark_set(mark_end, mark_start)
                self.text.mark_gravity(mark_end, tk.RIGHT)
                if display:
                    if MATH_PATTERN.search(display):
                        _render_rich_text(self.text, display, insert_pos=mark_start, text_tag=final_tag)
                    else:
                        self.text.insert(mark_start, display, final_tag)
                    self.text.insert(mark_end, "\n", final_tag)
                for mark in (mark_start, mark_end):
                    try:
                        self.text.mark_unset(mark)
                    except tk.TclError:
                        pass

            self._write(_replace)
            self.scroll_to_bottom()
            if on_done is not None:
                self.after(0, on_done)

        def _pump() -> None:
            if self._render_generation != generation:
                char_q.clear()
                pump_active[0] = False
                return
            if not char_q:
                pump_active[0] = False
                if llm_done[0]:
                    _do_replace()
                return
            ch = char_q.popleft()
            if not state["started"]:
                try:
                    s = self.text.index(mark_start)
                    e = self.text.index(mark_end)
                    def _clear_placeholder(start=s, end=e, m=mark_end) -> None:
                        self.text.delete(start, end)
                        self.text.mark_set(m, start)
                        self.text.mark_gravity(m, tk.RIGHT)

                    self._write(_clear_placeholder)
                except tk.TclError:
                    pass
                state["started"] = True
            self._write(lambda c=ch, t=final_tag, m=mark_end: self.text.insert(m, c, t))
            self.scroll_to_bottom()
            self.after(self._llm_char_speed_ms, _pump)

        def _append_token(token: str) -> None:
            if self._render_generation != generation or state["finished"] or not token:
                return
            streamed_any[0] = True
            for ch in token:
                char_q.append(ch)
            if not pump_active[0]:
                pump_active[0] = True
                _pump()

        def _finish(rendered: str | None) -> None:
            if self._render_generation != generation or state["finished"]:
                return
            state["finished"] = True
            display_ref[0] = (rendered or "").strip()
            llm_done[0] = True
            if display_ref[0] and not streamed_any[0]:
                for ch in display_ref[0]:
                    char_q.append(ch)
                if not pump_active[0]:
                    pump_active[0] = True
                    _pump()
                return
            if not pump_active[0] and not char_q:
                _do_replace()

        def _on_token(token: str) -> None:
            self.after(0, lambda t=token: _append_token(t))

        def _on_complete(rendered: str) -> None:
            self.after(0, lambda r=rendered: _finish(r))

        def _on_error(message: str) -> None:
            logger.debug("Description LLM image ignorée: %s", message)
            self.after(0, lambda: _finish(None))

        render_async(
            image_path=image_path,
            caption=caption,
            on_token=_on_token,
            on_complete=_on_complete,
            on_error=_on_error,
        )

    def _insert_streamed_math_fallback(self, block: dict, text: str, on_complete=None) -> None:
        def _insert() -> None:
            start = self.text.index("end-1c")
            self.text.insert(tk.END, text, "paragraph")
            end = self.text.index("end-1c")
            self.text.insert(tk.END, "\n", "paragraph")
            self._paragraph_ranges.append({
                "start": start,
                "end": end,
                "text": text,
            })

        self._write(_insert)
        self.embed_context_asset(block)
        self.scroll_to_bottom()
        if on_complete:
            on_complete(text)

    def render_completed_paragraph_with_llm(self, block: dict, on_complete=None) -> bool:
        if not self._paragraph_ranges or not _block_has_math(block):
            return False

        paragraph = self._paragraph_ranges[-1]
        text = (block.get("text") or paragraph.get("text") or "").strip()
        if not text:
            return False

        replace_range = (str(paragraph.get("start")), str(paragraph.get("end")))
        metadata = block.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["reader_math_streamed"] = True
        self._start_math_paragraph_render(
            block,
            text,
            replace_range=replace_range,
            on_complete=on_complete,
            insert_initial_text=False,
            append_newline_on_complete=False,
            paragraph_record=paragraph,
        )
        paragraph["text"] = text
        return True

    def _start_math_paragraph_render(
        self,
        block: dict,
        text: str,
        replace_range: tuple[str, str] | None,
        on_token=None,
        on_complete=None,
        placeholder_text: str | None = None,
        placeholder_tag: str = "math_loading",
        insert_initial_text: bool = True,
        append_newline_on_complete: bool = False,
        paragraph_record: dict | None = None,
    ) -> None:
        from llm.ollama_client import render_math_paragraph_stream_async

        metadata = block.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            block["metadata"] = metadata
        source_text = (text or "").strip()
        if source_text:
            metadata.setdefault("reader_source_text", source_text)
            metadata.setdefault("qa_source_text", source_text)

        idx = len(self._paragraph_ranges)
        mark_start = f"_mrender_s_{id(block)}_{idx}"
        mark_end = f"_mrender_e_{id(block)}_{idx}"
        generation = self._render_generation
        state = {"started": False, "finished": False, "finalized": False}

        def _insert_initial_render() -> None:
            if replace_range is not None:
                start, end = replace_range
                self.text.mark_set(mark_start, start)
                self.text.mark_gravity(mark_start, tk.LEFT)
                self.text.mark_set(mark_end, end)
                self.text.mark_gravity(mark_end, tk.RIGHT)
                return

            if placeholder_text is not None:
                pos = self.text.index("end-1c")
                self.text.mark_set(mark_start, pos)
                self.text.mark_gravity(mark_start, tk.LEFT)
                self.text.insert(tk.END, placeholder_text, placeholder_tag)
                self.text.mark_set(mark_end, "end-1c")
                self.text.mark_gravity(mark_end, tk.LEFT)
                self.text.insert(tk.END, "\n", placeholder_tag)
                return

            if insert_initial_text:
                pos = self.text.index("end-1c")
                self.text.mark_set(mark_start, pos)
                self.text.mark_gravity(mark_start, tk.LEFT)
                if MATH_PATTERN.search(text):
                    _render_rich_text(self.text, text, insert_pos=tk.END, text_tag="paragraph")
                else:
                    self.text.insert(tk.END, text, "paragraph")
                self.text.insert(tk.END, "\n", "paragraph")
                self.text.mark_set(mark_end, "end-1c")
                self.text.mark_gravity(mark_end, tk.RIGHT)

        self._write(_insert_initial_render)
        self.scroll_to_bottom()

        char_q: deque[str] = deque()
        pump_active = [False]
        llm_done = [False]
        display_ref: list[str] = [text]
        streamed_any = [False]
        streamed_parts: list[str] = []
        math_state = _new_incremental_math_state(f"mrender_{id(block)}_{idx}")

        def _finalize_display() -> None:
            if state["finalized"] or self._render_generation != generation:
                return
            state["finalized"] = True
            display = display_ref[0]

            def _replace() -> None:
                try:
                    start = self.text.index(mark_start)
                    end = self.text.index(mark_end)
                except tk.TclError:
                    return

                streamed_text = "".join(streamed_parts).strip()
                needs_final_math_render = bool(MATH_PATTERN.search(display) and not math_state["rendered_any"])
                needs_final_replacement = (
                    needs_final_math_render
                    or not state["started"]
                    or (streamed_text and streamed_text != display.strip())
                )
                if not needs_final_replacement:
                    paragraph_start = self.text.index(mark_start)
                    paragraph_end = self.text.index(mark_end)
                else:
                    self.text.delete(start, end)
                    self.text.mark_set(mark_end, mark_start)
                    self.text.mark_gravity(mark_end, tk.RIGHT)
                    if MATH_PATTERN.search(display):
                        _render_rich_text(self.text, display, insert_pos=mark_start, text_tag="paragraph")
                    else:
                        self.text.insert(mark_start, display, "paragraph")
                    paragraph_start = self.text.index(mark_start)
                    paragraph_end = self.text.index(mark_end)

                if append_newline_on_complete:
                    self.text.insert(mark_end, "\n", "paragraph")

                record = paragraph_record
                if record is not None:
                    record.update({"start": paragraph_start, "end": paragraph_end, "text": display})
                elif replace_range is None:
                    self._paragraph_ranges.append({
                        "start": paragraph_start,
                        "end": paragraph_end,
                        "text": display,
                    })

                for mark in (mark_start, mark_end):
                    try:
                        self.text.mark_unset(mark)
                    except tk.TclError:
                        pass

            self._write(_replace)
            self.embed_context_asset(block)
            self.scroll_to_bottom()
            if on_complete:
                on_complete(display)

        def _pump() -> None:
            if self._render_generation != generation:
                char_q.clear()
                pump_active[0] = False
                return
            if not char_q:
                pump_active[0] = False
                if llm_done[0]:
                    _finalize_display()
                return

            ch = char_q.popleft()
            if not state["started"]:
                try:
                    start = self.text.index(mark_start)
                    end = self.text.index(mark_end)
                    def _clear_placeholder(s=start, e=end, m=mark_end) -> None:
                        self.text.delete(s, e)
                        self.text.mark_set(m, s)
                        self.text.mark_gravity(m, tk.RIGHT)

                    self._write(_clear_placeholder)
                except tk.TclError:
                    pass
                state["started"] = True

            self._write(lambda c=ch, m=mark_end: _insert_incremental_math_char(
                self.text,
                c,
                insert_pos=m,
                text_tag="paragraph",
                state=math_state,
            ))
            self.scroll_to_bottom()
            if on_token:
                on_token(ch)
            self.after(self._llm_char_speed_ms, _pump)

        def _append_token(token: str) -> None:
            if self._render_generation != generation or state["finished"] or not token:
                return
            streamed_any[0] = True
            streamed_parts.append(token)
            for ch in token:
                char_q.append(ch)
            if not pump_active[0]:
                pump_active[0] = True
                _pump()

        def _finish(rendered: str | None) -> None:
            if self._render_generation != generation or state["finished"]:
                return
            state["finished"] = True
            display_ref[0] = (rendered or text).strip() or text
            metadata = block.get("metadata") or {}
            if isinstance(metadata, dict):
                metadata["reader_rendered_text"] = display_ref[0]
            block["text"] = display_ref[0]
            llm_done[0] = True
            if display_ref[0] and not streamed_any[0]:
                for ch in display_ref[0]:
                    char_q.append(ch)
                if not pump_active[0]:
                    pump_active[0] = True
                    _pump()
                return
            if not pump_active[0] and not char_q:
                _finalize_display()

        def _on_token(token: str) -> None:
            self.after(0, lambda t=token: _append_token(t))

        def _on_complete(rendered: str) -> None:
            self.after(0, lambda r=rendered: _finish(r))

        def _on_error(msg: str) -> None:
            logger.warning("Streaming LLM math échoué, repli texte brut: %s", msg)
            self.after(0, lambda: _finish(text))

        render_math_paragraph_stream_async(
            text,
            image_paths=self._math_image_paths(block),
            on_token=_on_token,
            on_complete=_on_complete,
            on_error=_on_error,
            document_context_before=self._collect_rendered_context_before(),
        )

    def _collect_rendered_context_before(self, max_chars: int = 600) -> str:
        """Return the last rendered paragraphs' text as LLM document context."""
        parts = []
        for record in reversed(self._paragraph_ranges[-6:]):
            t = (record.get("text") or "").strip()
            if t:
                parts.append(t)
        context = " … ".join(reversed(parts))
        return context[-max_chars:]

    def _math_image_paths(self, block: dict) -> list[str]:
        metadata = block.get("metadata") or {}
        context_path = metadata.get("context_asset_path") or block.get("context_asset_path")
        image_paths = []
        if block.get("type") == "formula":
            for path in (
                metadata.get("llm_crop_path"),
                metadata.get("formula_image_path"),
                block.get("image_path"),
            ):
                if path:
                    image_paths.append(str(path))
        skip_context = _context_asset_is_unsafe_for_math_render(block)
        if context_path and not skip_context:
            image_paths.append(str(context_path))
        image_paths.extend(str(path) for path in metadata.get("math_dense_context_assets") or [] if path)
        for asset in metadata.get("llm_assets") or []:
            if isinstance(asset, dict) and asset.get("type") == "image" and asset.get("path"):
                path = str(asset.get("path"))
                if skip_context and context_path and path == str(context_path):
                    continue
                image_paths.append(path)
        return list(dict.fromkeys(image_paths))

    def _insert_bullet_list(self, block: dict) -> None:
        items = [str(item).strip() for item in block.get("items") or [] if str(item).strip()]
        text = "\n".join(f"• {item}" for item in items) if items else (block.get("text") or "").strip()
        if not text:
            return

        def _insert() -> None:
            start = self.text.index("end-1c")
            _render_rich_text(self.text, text, insert_pos=tk.END, text_tag="bullet_list")
            end = self.text.index("end-1c")
            self.text.insert(tk.END, "\n", "bullet_list")
            self._paragraph_ranges.append({
                "start": start,
                "end": end,
                "text": text,
            })

        self._write(_insert)
        self.embed_context_asset(block)

    def embed_context_asset(self, block: dict) -> bool:
        if not _should_show_context_asset(block):
            return False
        metadata = block.get("metadata") or {}
        image_path = metadata.get("context_asset_path") or block.get("context_asset_path")
        if not image_path:
            return False
        return self._insert_image_file(str(image_path), caption_display=False, max_height=_MAX_READER_CONTEXT_HEIGHT)

    def embed_slide_page(self, page_number: int, on_analysis_complete=None) -> None:
        """Render a PDF slide page full-width and stream an LLM analysis below it."""
        self.hide_loading_overlay()
        if not self._pdf_path or not Path(self._pdf_path).exists():
            if on_analysis_complete is not None:
                self.after(0, on_analysis_complete)
            return
        try:
            import fitz
            from PIL import Image, ImageTk

            tmp_dir = Path(self._pdf_path).parent / ".slide_cache"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"slide_{abs(hash(self._pdf_path))}_{page_number}.png"

            with fitz.open(self._pdf_path) as doc:
                if page_number - 1 >= len(doc):
                    return
                page = doc[page_number - 1]
                page_width = float(page.rect.width)

                # High-res crop for LLM (2×)
                if not tmp_path.exists():
                    pix_hires = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    pix_hires.save(str(tmp_path))

                # Display render scaled to fill the reading area width
                available_width = self.text.winfo_width() - 88
                if available_width <= 8:
                    available_width = 820
                scale = available_width / max(1.0, page_width)
                scale = min(scale, 3.0)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            photo = ImageTk.PhotoImage(image)
            self._image_refs.append(photo)
            self._write(lambda i=photo: (
                self.text.insert(tk.END, "\n", "media"),
                self.text.image_create(tk.END, image=i, padx=0, pady=4),
                self.text.insert(tk.END, "\n", "media"),
            ))
            self.scroll_to_bottom()

            from llm.ollama_client import render_slide_stream_async
            self._stream_image_description(
                block={},
                image_path=str(tmp_path),
                caption="",
                loading_text="Analyse de la slide…",
                loading_tag="schema_loading",
                final_tag="schema_description",
                render_async=render_slide_stream_async,
                on_done=on_analysis_complete,
            )
        except Exception as exc:
            logger.warning("Rendu slide p.%s impossible : %s", page_number, exc)
            if on_analysis_complete is not None:
                self.after(0, on_analysis_complete)

    def _insert_formula(self, block: dict) -> None:
        metadata = block.get("metadata") or {}
        if metadata.get("render_mode") == "pdf_crop":
            if self._insert_formula_image_path(block):
                return
            if self._insert_formula_pdf_crop(block):
                return
            if _formula_should_render_with_llm(block):
                self._start_math_paragraph_render(
                    block,
                    _formula_source_text(block),
                    replace_range=None,
                    placeholder_text=t("reading.processing"),
                    insert_initial_text=False,
                    append_newline_on_complete=True,
                )
                return
            fallback = (block.get("latex") or block.get("text") or "").strip()
            if fallback:
                self._write(lambda t=fallback: self.text.insert(tk.END, "\n" + t + "\n", "paragraph"))
            else:
                self._write(lambda: self.text.insert(tk.END, "\n[formule]\n", "note"))
            return

        latex = (block.get("latex") or "").strip()
        display = block.get("display", True)
        if latex:
            try:
                from core.latex import formula_to_tk_image
                max_height = _MAX_READER_FORMULA_HEIGHT if display else _MAX_READER_INLINE_FORMULA_HEIGHT
                img = formula_to_tk_image(latex, display, max_height=max_height)
                if img:
                    self._image_refs.append(img)
                    self._write(lambda i=img: (
                        self.text.insert(tk.END, "\n"),
                        self.text.image_create(tk.END, image=i, padx=8, pady=6),
                        self.text.insert(tk.END, "\n"),
                    ))
                    return
            except Exception as exc:
                logger.error("Rendu formule échoué : %s", exc)
            self._write(lambda t=latex: self.text.insert(tk.END, "\n" + t + "\n", "paragraph"))
            return

        text = (block.get("text") or "").strip()
        if text:
            self._write(lambda t=text: self.text.insert(tk.END, "\n" + t + "\n", "paragraph"))
        else:
            self._write(lambda: self.text.insert(tk.END, "\n[formule]\n", "note"))

    def _insert_formula_image_path(self, block: dict) -> bool:
        image_path = block.get("image_path") or (block.get("metadata") or {}).get("formula_image_path")
        if not image_path:
            return False
        return self._insert_image_file(
            str(image_path),
            caption_display=False,
            max_width=_MAX_READER_FORMULA_WIDTH,
            max_height=_MAX_READER_FORMULA_CROP_HEIGHT,
            trim_whitespace=True,
        )

    def _insert_formula_pdf_crop(self, block: dict) -> bool:
        if not self._pdf_path or not Path(self._pdf_path).exists():
            return False

        bbox = block.get("bbox")
        page_number = block.get("page_number") or block.get("page") or block.get("page_start")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or page_number is None:
            return False

        try:
            import fitz  # type: ignore
            from PIL import Image

            page_index = max(0, int(page_number) - 1)
            with fitz.open(self._pdf_path) as doc:
                if page_index >= len(doc):
                    return False
                page = doc[page_index]
                rect = fitz.Rect(*(float(value) for value in bbox[:4]))
                padding = 12.0
                rect = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)
                rect = fitz.Rect(
                    max(rect.x0, page.rect.x0),
                    max(rect.y0, page.rect.y0),
                    min(rect.x1, page.rect.x1),
                    min(rect.y1, page.rect.y1),
                )
                if rect.is_empty or rect.width <= 1 or rect.height <= 1:
                    return False
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            image = _trim_light_background(image, padding=10)
            photo = self._photo_from_image(
                image,
                max_width=_MAX_READER_FORMULA_WIDTH,
                max_height=_MAX_READER_FORMULA_CROP_HEIGHT,
            )
            self._insert_photo(photo)
            return True
        except Exception as exc:
            logger.warning("Crop PDF formule indisponible : %s", exc)
            return False

    def _insert_table(self, block: dict) -> None:
        metadata = block.get("metadata") or {}
        caption = (block.get("caption") or "").strip()
        caption_display = block.get("caption_display", metadata.get("caption_display", True))
        caption_inserted = [False]

        def _insert_caption() -> None:
            if caption_inserted[0]:
                return
            if caption and caption_display is not False:
                self._write(lambda c=caption: self.text.insert(tk.END, "\n" + c + "\n", "caption"))
                caption_inserted[0] = True

        table_image_path = metadata.get("table_image_path")
        if table_image_path:
            _insert_caption()
        if table_image_path and self._insert_image_file(str(table_image_path), caption_display=False, max_height=_MAX_READER_TABLE_HEIGHT):
            self._stream_table_description(block)
            return

        markdown = (block.get("markdown") or "").strip()
        rendered = _markdown_table_to_monospace(markdown)
        if rendered:
            _insert_caption()
            self._write(lambda t=rendered: self.text.insert(tk.END, "\n" + t + "\n", "table"))
            if table_image_path:
                self._stream_table_description(block)
            return

        text = (block.get("text") or "").strip()
        if text:
            _insert_caption()
            fallback = "[Tableau]\n" + text
            self._write(lambda t=fallback: self.text.insert(tk.END, "\n" + t + "\n", "paragraph"))
            if table_image_path:
                self._stream_table_description(block)
            return

        self._write(lambda: self.text.insert(tk.END, "\n[Tableau]\n", "note"))

    def _insert_figure(self, block: dict) -> None:
        caption = block.get("caption", "")
        image_path = block.get("image_path")
        displayed = False
        if image_path:
            displayed = self._insert_image_file(str(image_path), caption_display=False, max_height=_MAX_READER_FIGURE_HEIGHT)
        if not displayed:
            crop_path = self._insert_figure_pdf_crop(block)
            if crop_path:
                block["image_path"] = crop_path
                meta = block.get("metadata")
                if not isinstance(meta, dict):
                    block["metadata"] = {}
                block["metadata"]["pdf_cropped"] = True
        caption_display = block.get("caption_display", (block.get("metadata") or {}).get("caption_display", True))
        if caption and caption_display is not False:
            self._write(lambda c=caption: self.text.insert(tk.END, c + "\n", "caption"))

    def _insert_figure_pdf_crop(self, block: dict) -> str | None:
        """Crop figure region directly from PDF when image_path is missing or invalid."""
        if not self._pdf_path or not Path(self._pdf_path).exists():
            return None
        bbox = block.get("bbox")
        page_number = block.get("page_number") or block.get("page") or block.get("page_start")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or page_number is None:
            return None
        try:
            import fitz
            from PIL import Image

            page_index = max(0, int(page_number) - 1)
            with fitz.open(self._pdf_path) as doc:
                if page_index >= len(doc):
                    return None
                page = doc[page_index]
                rect = fitz.Rect(*(float(v) for v in bbox[:4]))
                padding = 8.0
                rect = fitz.Rect(
                    max(rect.x0 - padding, page.rect.x0),
                    max(rect.y0 - padding, page.rect.y0),
                    min(rect.x1 + padding, page.rect.x1),
                    min(rect.y1 + padding, page.rect.y1),
                )
                if rect.is_empty or rect.width <= 10 or rect.height <= 10:
                    return None
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            crops_dir = Path(self._pdf_path).parent / ".figure_crops"
            crops_dir.mkdir(exist_ok=True)
            block_id = str(block.get("id") or f"p{page_number}_{id(block)}")
            safe_id = re.sub(r"[^\w]", "_", block_id)[:60]
            crop_path = crops_dir / f"{safe_id}.png"
            image.save(str(crop_path))

            photo = self._photo_from_image(
                image,
                max_width=self._available_media_width(),
                max_height=_MAX_READER_FIGURE_CROP_HEIGHT,
            )
            self._insert_photo(photo)
            return str(crop_path)
        except Exception as exc:
            logger.warning("Crop PDF figure indisponible : %s", exc)
            return None

    def _insert_image_file(
        self,
        image_path: str,
        caption_display: bool = True,
        caption: str = "",
        max_width: int | None = None,
        max_height: int | None = None,
        trim_whitespace: bool = False,
    ) -> bool:
        resolved = self._resolve_image_path(image_path)
        if resolved is None:
            return False

        try:
            from PIL import Image

            with Image.open(resolved) as opened:
                image = opened.copy()
            if trim_whitespace:
                image = _trim_light_background(image, padding=10)
            photo = self._photo_from_image(image, max_width=max_width, max_height=max_height)
            self._insert_photo(photo)
            if caption and caption_display:
                self._write(lambda c=caption: self.text.insert(tk.END, c + "\n", "caption"))
            return True
        except Exception as exc:
            logger.warning("Impossible de charger l'image %s : %s", image_path, exc)
            return False

    def _photo_from_image(self, image, max_width: int | None = None, max_height: int | None = None):
        from PIL import Image, ImageTk

        available_width = self._available_media_width()
        target_width = min(available_width, max_width) if max_width else available_width
        width_ratio = target_width / max(image.width, 1)
        height_ratio = (max_height / max(image.height, 1)) if max_height else 1.0
        ratio = min(1.0, width_ratio, height_ratio)
        if ratio < 1.0:
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), resample)
        photo = ImageTk.PhotoImage(image)
        self._image_refs.append(photo)
        return photo

    def _insert_photo(self, photo) -> None:
        self._write(lambda i=photo: (
            self.text.insert(tk.END, "\n", "media"),
            self.text.image_create(tk.END, image=i, padx=8, pady=6),
            self.text.insert(tk.END, "\n", "media"),
        ))

    def _available_media_width(self) -> int:
        width = self.text.winfo_width()
        if width <= 1:
            width = 860
        return max(160, min(_MAX_READER_MEDIA_WIDTH, width - 130))

    def _resolve_image_path(self, image_path: str) -> Path | None:
        raw = Path(str(image_path)).expanduser()
        candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw]
        if self._pdf_path:
            candidates.append(Path(self._pdf_path).parent / raw)
        if not raw.is_absolute():
            asset_root = Path(ASSETS_DIR)
            package_root = asset_root.parent
            project_root = package_root.parent
            candidates.extend([package_root / raw, project_root / raw])
            parts = raw.parts
            if parts and parts[0] == "assets":
                candidates.append(asset_root / Path(*parts[1:]))
            if len(parts) >= 2 and parts[0] == "nwol" and parts[1] == "assets":
                candidates.append(asset_root / Path(*parts[2:]))
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
        return None

    def _insert_embedded_frame(self, frame: tk.Frame) -> None:
        self._embedded_frames.append(frame)
        self._resize_frame(frame)
        self._write(lambda f=frame: (
            self.text.insert(tk.END, "\n"),
            self.text.window_create(tk.END, window=f),
            self.text.insert(tk.END, "\n"),
        ))
        self.scroll_to_bottom()

    def _resize_embedded_frames(self, _event=None) -> None:
        alive_frames: list[tk.Frame] = []
        for frame in self._embedded_frames:
            try:
                if not frame.winfo_exists():
                    continue
            except tk.TclError:
                continue
            alive_frames.append(frame)
            self._resize_frame(frame)
        self._embedded_frames = alive_frames

    def _resize_frame(self, frame: tk.Frame) -> None:
        width = max(420, min(_MAX_EMBEDDED_FRAME_WIDTH, self.text.winfo_width() - 92))
        try:
            frame.configure(width=width)
            if getattr(frame, "_nwol_fill_reader_width", False):
                _lock_frame_requested_width(frame, width)
        except tk.TclError:
            pass

    def _type_text(self, text_var: tk.StringVar, text: str, index: int = 0) -> None:
        if index >= len(text):
            return
        text_var.set(text[:index + 1])
        self.scroll_to_bottom()
        self.after(8, lambda: self._type_text(text_var, text, index + 1))

    def _write(self, fn) -> None:
        self.text.configure(state=tk.NORMAL)
        fn()
        self.text.configure(state=tk.DISABLED)


class _ParagraphRephraseWidget(tk.Frame):
    """Bouton ↻ Reformuler ancré au bas-droit du paragraphe parent.

    Trois états : idle (bouton) → loading → done (reformulation).
    Toujours associé au bloc capturé à l'insertion, indépendamment
    de la position de lecture courante.
    """

    _IDLE_HEIGHT = 26

    def __init__(self, master, block: dict, on_rephrase, **kwargs):
        super().__init__(master, bg=theme.SURFACE, **kwargs)
        self._block = block
        self._on_rephrase = on_rephrase
        self.pack_propagate(False)
        self.configure(height=self._IDLE_HEIGHT)
        self._build_idle()

    def _clear(self) -> None:
        for w in self.winfo_children():
            w.destroy()

    def _build_idle(self) -> None:
        self._clear()
        self.pack_propagate(False)
        self.configure(height=self._IDLE_HEIGHT)
        btn = tk.Button(
            self,
            text=t("qa.rephrase_btn"),
            command=self._on_click,
            bg=theme.SURFACE,
            fg=theme.MUTED,
            activebackground=theme.SURFACE_SOFT,
            activeforeground=theme.TEXT_SOFT,
            font=(theme.FONT_UI, 9),
            relief="flat",
            cursor="hand2",
            bd=0,
            highlightthickness=0,
            takefocus=0,
            padx=6,
            pady=2,
        )
        # place() permet le positionnement bas-droite indépendamment de la
        # largeur du frame fixée par _resize_frame
        btn.place(relx=1.0, rely=0.5, anchor="e", x=-8)

    def _on_click(self) -> None:
        self._clear()
        lbl = tk.Label(
            self,
            text=t("qa.rephrase_loading"),
            bg=theme.SURFACE,
            fg=theme.MUTED,
            font=(theme.FONT_UI, 9, "italic"),
        )
        lbl.place(relx=1.0, rely=0.5, anchor="e", x=-8)
        self._on_rephrase(self._block, self._show_result, self._show_error)

    def _show_result(self, rephrasing: dict) -> None:
        try:
            self.after(0, lambda r=rephrasing: self._render_result(r))
        except tk.TclError:
            pass

    def _render_result(self, rephrasing: dict) -> None:
        self._clear()
        self.pack_propagate(True)
        outer = tk.Frame(
            self,
            bg=theme.WARNING_SOFT,
            highlightthickness=1,
            highlightbackground=theme.WARNING,
            highlightcolor=theme.WARNING,
        )
        outer.pack(fill="x", padx=(8, 8), pady=(0, 6))
        stripe = tk.Frame(outer, bg=theme.WARNING, width=5)
        stripe.pack(side="left", fill="y")
        content = tk.Frame(outer, bg=theme.WARNING_SOFT)
        content.pack(side="left", fill="both", expand=True)
        tk.Label(
            content,
            text=f"↻ Vu autrement : {rephrasing.get('rephrasing_angle', '')}",
            bg=theme.WARNING_SOFT,
            fg=theme.WARNING,
            font=(theme.FONT_UI, 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 0))
        _rich_text_widget(
            content,
            _normalize_llm_text(rephrasing.get("rephrased_paragraph", "")),
            bg=theme.WARNING_SOFT,
            fg=theme.TEXT,
            font=(theme.FONT_UI, 10),
            justify="left",
        ).pack(fill="x", padx=14, pady=(6, 0))
        note = rephrasing.get("note")
        if note:
            _rich_text_widget(
                content,
                f"Note : {_normalize_llm_text(note)}",
                bg=theme.WARNING_SOFT,
                fg=theme.WARNING,
                font=(theme.FONT_UI, 9, "italic"),
                justify="left",
            ).pack(fill="x", padx=14, pady=(4, 10))
        else:
            tk.Frame(content, bg=theme.WARNING_SOFT, height=10).pack()

    def _show_error(self, message: str) -> None:
        try:
            self.after(0, lambda m=message: self._render_error(m))
        except tk.TclError:
            pass

    def _render_error(self, message: str) -> None:
        self._clear()
        self.pack_propagate(False)
        self.configure(height=self._IDLE_HEIGHT)
        lbl = tk.Label(
            self,
            text=f"Reformulation indisponible : {message[:80]}",
            bg=theme.SURFACE,
            fg=theme.MUTED,
            font=(theme.FONT_UI, 9, "italic"),
        )
        lbl.place(relx=1.0, rely=0.5, anchor="e", x=-8)

    def _type_text(self, var: tk.StringVar, text: str, index: int = 0) -> None:
        if index >= len(text):
            return
        try:
            var.set(text[:index + 1])
            self.after(8, lambda: self._type_text(var, text, index + 1))
        except tk.TclError:
            pass


def _callout_frame(master, label: str, text: str, bg: str, fg: str) -> tk.Frame:
    frame = _message_frame(master, bg, fg)
    content = frame._content
    tk.Label(
        content,
        text=label,
        bg=bg,
        fg=fg,
        font=(theme.FONT_UI, 10, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(10, 2))
    _rich_text_widget(
        content,
        text,
        bg=bg,
        fg=theme.TEXT,
        font=(theme.FONT_UI, 10),
    ).pack(fill="x", padx=14, pady=(0, 10))
    return frame


def _message_frame(master, bg: str, border: str) -> tk.Frame:
    frame = tk.Frame(
        master,
        bg=bg,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
    )
    frame._nwol_fill_reader_width = True
    stripe = tk.Frame(frame, bg=border, width=5)
    stripe.pack(side="left", fill="y")
    frame._content = tk.Frame(frame, bg=bg)
    frame._content.pack(side="left", fill="both", expand=True)
    return frame


def _lock_frame_requested_width(frame: tk.Frame, width: int) -> None:
    """Make Text-embedded message frames use the reader width, not child width."""
    try:
        frame.update_idletasks()
        height = max(frame.winfo_reqheight(), frame.winfo_height(), 1)
        frame.configure(width=width, height=height)
        frame.pack_propagate(False)
        content = getattr(frame, "_content", None)
        if content is not None:
            content.configure(width=max(1, width - 7))
    except tk.TclError:
        pass


_MAX_CELL_WIDTH = 28


def _wrap_cell(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    return [text[i:i + width] for i in range(0, len(text), width)]


def _markdown_table_to_monospace(markdown: str) -> str:
    rows = _parse_markdown_table(markdown)
    if not rows:
        return ""
    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]
    widths = [min(_MAX_CELL_WIDTH, max(len(row[index]) for row in rows)) for index in range(column_count)]

    def border() -> str:
        return "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def row_lines(row: list[str]) -> list[str]:
        wrapped = [_wrap_cell(row[i], widths[i]) for i in range(len(row))]
        height = max(len(w) for w in wrapped)
        lines = []
        for sub in range(height):
            cells = [(wrapped[i][sub] if sub < len(wrapped[i]) else "").ljust(widths[i]) for i in range(len(row))]
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    lines = [border()]
    lines.extend(row_lines(rows[0]))
    lines.append(border())
    for row in rows[1:]:
        lines.extend(row_lines(row))
    lines.append(border())
    return "\n".join(lines)


def _parse_markdown_table(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)
    return rows


def _should_show_context_asset(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if metadata.get("context_asset_display") is False:
        return False
    image_path = metadata.get("context_asset_path") or block.get("context_asset_path")
    if not image_path:
        return False
    reason = metadata.get("context_asset_reason")
    if metadata.get("render_mode") == "context_crop_only":
        return True
    if metadata.get("formula_mode") == "ambiguous":
        return True
    if reason == "math_dense_text":
        return False
    if reason == "inline_math":
        return bool(metadata.get("context_asset_display"))
    if metadata.get("context_asset_display") is True:
        return True
    if reason == "fragmented_math_text":
        return True
    if metadata.get("reader_render_mode") == "context_crop_only":
        return True
    if metadata.get("render_mode") == "text_with_context_crop" and reason == "low_confidence_text":
        return True
    if metadata.get("render_mode") == "context_crop_only":
        return True
    return metadata.get("render_mode") == "text_with_context_crop"


def _should_replace_text_with_context_asset(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    return metadata.get("render_mode") == "context_crop_only" or metadata.get("reader_render_mode") == "context_crop_only"


def _formula_should_render_with_llm(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if metadata.get("reader_disable_latex_llm"):
        return False
    if metadata.get("formula_mode") != "display":
        return False
    if block.get("image_path") or metadata.get("formula_image_path") or metadata.get("llm_crop_path"):
        return False
    if metadata.get("latex_corrupt") or latex_looks_corrupt(block.get("latex") or block.get("text")):
        return True
    return bool(
        metadata.get("wide_initial_crop")
        or metadata.get("needs_latex_llm")
    )


def _formula_source_text(block: dict) -> str:
    text = str(block.get("latex") or block.get("text") or "").strip()
    if text:
        metadata = block.get("metadata") or {}
        if metadata.get("latex_corrupt") or latex_looks_corrupt(text):
            return t("reading.formula_image_prompt")
        stripped = text.strip()
        if stripped.startswith("$$") and stripped.endswith("$$"):
            return stripped
        inner = stripped[1:-1].strip() if stripped.startswith("$") and stripped.endswith("$") else stripped
        return f"$${inner}$$"
    return t("reading.formula_image_prompt")


def _context_asset_is_unsafe_for_math_render(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    reason = metadata.get("context_asset_reason")
    if reason not in {"inline_math", "math_dense_text"}:
        return False
    if metadata.get("mixed_columns_risk"):
        return True

    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        width = abs(float(bbox[2]) - float(bbox[0]))
        page_width = float(metadata.get("page_width") or 0.0)
    except (TypeError, ValueError):
        return False
    if page_width > 0.0:
        return width >= page_width * 0.62
    return width >= 430.0


def _block_render_image_path(block: dict, preferred_metadata_key: str | None = None) -> str | None:
    metadata = block.get("metadata") or {}
    if preferred_metadata_key and metadata.get(preferred_metadata_key):
        return str(metadata.get(preferred_metadata_key))

    for asset in metadata.get("llm_assets") or []:
        if isinstance(asset, dict) and asset.get("type") == "image" and asset.get("path"):
            return str(asset.get("path"))

    image_path = block.get("image_path")
    return str(image_path) if image_path else None


def _normalize_llm_text(text: str) -> str:
    """Convertit les séquences d'échappement littérales que le LLM retourne parfois."""
    if not isinstance(text, str):
        return str(text or "")
    # \n et \t littéraux (deux caractères) → vrais caractères de contrôle
    return text.replace("\\n", "\n").replace("\\t", "\t")


def _trim_light_background(image, *, padding: int = 8):
    """Trim mostly-white margins from PDF formula crops while keeping antialiasing."""
    try:
        from PIL import Image

        source = image.convert("RGBA") if image.mode not in {"RGB", "RGBA", "L"} else image
        gray = source.convert("L")
        mask = gray.point(lambda pixel: 255 if pixel < 248 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return image

        left, top, right, bottom = bbox
        if right - left < 2 or bottom - top < 2:
            return image
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)
        if left == 0 and top == 0 and right == image.width and bottom == image.height:
            return image
        return image.crop((left, top, right, bottom))
    except Exception:
        return image
