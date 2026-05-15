# ui/gauges_panel.py — Bande de jauges temps réel
from __future__ import annotations

import tkinter as tk

from i18n import t
from ui import theme
from ui.top_nav import Tooltip

BG = theme.SURFACE
TEXT = theme.TEXT
MUTED = theme.MUTED
BAR_BG = theme.BORDER
BAR_FG = theme.ACCENT

GAUGE_LABELS = {
    "attention":             ("◉",  "gauges.attention"),
    "context_comprehension": ("Aa", "gauges.context_comprehension"),
    "creativity":            ("✦",  "gauges.creativity"),
    "retention":             ("◇",  "gauges.retention"),
    "curiosity":             ("?",  "gauges.curiosity"),
    "meta_cognition":        ("MC", "gauges.meta_cognition"),
}


class GaugesPanel(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            bg=BG,
            width=220,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            **kwargs,
        )
        self.pack_propagate(False)
        self._bars: dict[str, tuple[tk.Canvas, int, tk.Label]] = {}
        self._values: dict[str, float] = {}
        self._subject_sep: tk.Widget | None = None
        self._subject_hdr: tk.Widget | None = None
        self._subject_frame: tk.Widget | None = None
        self._subject_info: tuple[str, str] | None = None  # (subject, display_label)
        self._build()

    def _build(self) -> None:
        tk.Label(
            self,
            text=t("gauges.title"),
            bg=BG,
            fg=TEXT,
            font=(theme.FONT_UI, 12, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(
            self,
            text=t("gauges.subtitle"),
            bg=BG,
            fg=MUTED,
            font=(theme.FONT_UI, 9),
        ).pack(anchor="w", padx=18, pady=(0, 10))

        for name, (icon, label_key) in GAUGE_LABELS.items():
            full = t(label_key)
            frame = tk.Frame(self, bg=BG)
            frame.pack(fill="x", padx=18, pady=10)

            top = tk.Frame(frame, bg=BG)
            top.pack(fill="x")
            tk.Label(top, text=icon, bg=BG, fg=theme.ACCENT, font=(theme.FONT_UI, 10, "bold"), width=3, anchor="w").pack(side="left")
            tk.Label(top, text=full, bg=BG, fg=TEXT, font=(theme.FONT_UI, 10, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
            value_lbl = tk.Label(top, text="50", bg=BG, fg=MUTED, font=(theme.FONT_UI, 10, "bold"), width=3)
            value_lbl.pack(side="right")

            canvas = tk.Canvas(frame, height=16, bg=BG, highlightthickness=0)
            canvas.pack(fill="x", pady=(5, 0))
            theme.draw_pill_bar(canvas, 160, 10, 50, fill=BAR_FG, background=BAR_BG, tag="bar")

            self._values[name] = 50.0

            def _resize(event, c=canvas, gauge=name):
                value = self._values.get(gauge, 50.0)
                theme.draw_pill_bar(c, event.width, 10, value, fill=_color_for_value(value), background=BAR_BG, tag="bar")

            canvas.bind("<Configure>", _resize)
            Tooltip(frame, t("gauges.gauge_tip", label=full))
            self._bars[name] = (canvas, 0, value_lbl)

    def update_gauge(self, name: str, value: float) -> None:
        if name not in self._bars:
            return
        target = _clamp(value)
        start = _clamp(self._values.get(name, target))
        canvas, fg_id, label = self._bars[name]

        def _paint(current: float) -> None:
            self._values[name] = current
            width = max(1, canvas.winfo_width())
            theme.draw_pill_bar(canvas, width, 10, current, fill=_color_for_value(current), background=BAR_BG, tag="bar")
            label.configure(text=f"{int(round(current))}")

        if abs(target - start) < 0.5:
            _paint(target)
            return

        def _update(progress: float) -> None:
            eased = theme.ease_out_cubic(progress)
            _paint(start + (target - start) * eased)

        theme.animate(canvas, theme.ANIM_NORMAL, _update, lambda: _paint(target))

    def update_all(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.update_gauge(name, value)

    def add_subject_gauge(self, subject: str, label: str, value: float = 50.0) -> None:
        self.remove_subject_gauge()

        sep = tk.Frame(self, bg=theme.BORDER, height=1)
        sep.pack(fill="x", padx=18, pady=(8, 0))
        self._subject_sep = sep

        hdr = tk.Label(self, text=t("gauges.subject_header"), bg=BG, fg=MUTED, font=(theme.FONT_UI, 9))
        hdr.pack(anchor="w", padx=18, pady=(6, 2))
        self._subject_hdr = hdr

        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="x", padx=18, pady=8)
        self._subject_frame = frame

        top = tk.Frame(frame, bg=BG)
        top.pack(fill="x")
        tk.Label(top, text="S", bg=BG, fg=theme.ACCENT, font=(theme.FONT_UI, 10, "bold"), width=3, anchor="w").pack(side="left")
        tk.Label(top, text=label, bg=BG, fg=TEXT, font=(theme.FONT_UI, 10, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
        value_lbl = tk.Label(top, text=str(int(round(value))), bg=BG, fg=MUTED, font=(theme.FONT_UI, 10, "bold"), width=3)
        value_lbl.pack(side="right")

        canvas = tk.Canvas(frame, height=16, bg=BG, highlightthickness=0)
        canvas.pack(fill="x", pady=(5, 0))
        theme.draw_pill_bar(canvas, 160, 10, value, fill=_color_for_value(value), background=BAR_BG, tag="bar")

        self._values["subject"] = value

        def _resize(event, c=canvas):
            v = self._values.get("subject", 50.0)
            theme.draw_pill_bar(c, event.width, 10, v, fill=_color_for_value(v), background=BAR_BG, tag="bar")

        canvas.bind("<Configure>", _resize)
        Tooltip(frame, t("gauges.subject_tip", label=label))
        self._bars["subject"] = (canvas, 0, value_lbl)
        self._subject_info = (subject, label)

    def remove_subject_gauge(self) -> None:
        for attr in ("_subject_sep", "_subject_hdr", "_subject_frame"):
            widget = getattr(self, attr, None)
            if widget:
                widget.destroy()
                setattr(self, attr, None)
        self._bars.pop("subject", None)
        self._values.pop("subject", None)
        self._subject_info = None

    def refresh_lang(self) -> None:
        saved_values = dict(self._values)
        subject_info = self._subject_info
        for child in self.winfo_children():
            child.destroy()
        self._bars = {}
        self._values = {}
        self._subject_sep = None
        self._subject_hdr = None
        self._subject_frame = None
        self._subject_info = None
        self._build()
        for name, value in saved_values.items():
            if name != "subject":
                self._values[name] = value
        if subject_info:
            _subject, label = subject_info
            cur_val = saved_values.get("subject", 50.0)
            self.add_subject_gauge(_subject, label, cur_val)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _color_for_value(value: float) -> str:
    if value >= 75:
        return theme.SUCCESS
    if value >= 45:
        return theme.ACCENT
    return theme.WARNING
