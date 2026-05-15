# ui/chapter_selector.py — Sas de sélection chapitre / sous-sections
from __future__ import annotations
import tkinter as tk

from core.chapter_navigation import child_sections_for_chapter
from core.document import normalize_chapter_list
from i18n import t
from ui import theme
from ui.top_nav import Tooltip

BG          = theme.BG
TEXT_DARK   = theme.TEXT
TEXT_MUTED  = theme.MUTED
ACCENT      = theme.ACCENT
SEL_BG      = theme.ACCENT_SOFT_HOVER


class ChapterSelector(tk.Frame):
    """Affiché après l'import d'un PDF : choix chapitre → section détaillée → étudier."""

    def __init__(self, parent, on_study, on_back, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._on_study = on_study
        self._on_back  = on_back
        self._chapters: list[dict] = []
        self._level1:   list[dict] = []
        self._subs:     list[dict] = []
        self._selected: dict | None = None
        self._filename: str = ""
        self._build()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def load(self, filename: str, chapters: list[dict]) -> None:
        self._filename = filename
        self._chapters = normalize_chapter_list(chapters)
        self._file_lbl.configure(text=filename)

        self._level1 = [c for c in self._chapters if c.get("toc_level", 1) == 1] or self._chapters

        self._ch_list.delete(0, tk.END)
        for ch in self._level1:
            self._ch_list.insert(tk.END, f"  {ch['title']}")

        self._sub_list.delete(0, tk.END)
        self._sub_hint.configure(text=t("chapter.hint"))
        self._selected = None
        self._study_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        # --- Barre supérieure ---
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=56, pady=(34, 0))

        back_btn = theme.make_button(
            top, text=t("chapter.back"),
            kind="ghost",
            command=self._on_back,
            font=(theme.FONT_UI, 11, "bold"),
        )
        back_btn.pack(side="left")
        Tooltip(back_btn, t("chapter.back_tip"))

        tk.Label(
            top, text="MetaC-App",
            font=(theme.FONT_TITLE, 18, "bold"),
            bg=BG, fg=TEXT_DARK,
        ).pack(side="right")

        # --- Titre ---
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="x", padx=56, pady=(30, 0))

        tk.Label(
            mid, text=t("chapter.title"),
            font=(theme.FONT_TITLE, 27, "bold"),
            bg=BG, fg=TEXT_DARK,
        ).pack(anchor="w")

        self._file_lbl = tk.Label(
            mid, text="",
            font=(theme.FONT_UI, 11),
            bg=BG, fg=TEXT_MUTED,
        )
        self._file_lbl.pack(anchor="w", pady=(4, 0))

        theme.divider(self, padx=56, pady=(22, 26))

        # --- Zone centrale : deux colonnes ---
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=56)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)

        # Titres colonnes
        tk.Label(
            content, text=t("chapter.chapters"),
            font=(theme.FONT_UI, 12, "bold"),
            bg=BG, fg=TEXT_DARK,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        tk.Label(
            content, text=t("chapter.subsections"),
            font=(theme.FONT_UI, 12, "bold"),
            bg=BG, fg=TEXT_DARK,
        ).grid(row=0, column=1, sticky="w", padx=(24, 0), pady=(0, 8))

        # Liste chapitres
        ch_frame = theme.surface_frame(content)
        ch_frame.grid(row=1, column=0, sticky="nsew")
        ch_scroll = tk.Scrollbar(ch_frame)
        ch_scroll.pack(side="right", fill="y")

        self._ch_list = theme.style_listbox(tk.Listbox(
            ch_frame,
            font=(theme.FONT_UI, 12),
            activestyle="none",
            yscrollcommand=ch_scroll.set,
        ))
        ch_scroll.configure(command=self._ch_list.yview)
        self._ch_list.pack(fill="both", expand=True, padx=1, pady=1)
        self._ch_list.bind("<<ListboxSelect>>", self._on_chapter_click)

        # Liste sous-chapitres
        sub_frame = theme.surface_frame(content)
        sub_frame.grid(row=1, column=1, sticky="nsew", padx=(24, 0))
        sub_scroll = tk.Scrollbar(sub_frame)
        sub_scroll.pack(side="right", fill="y")

        self._sub_list = theme.style_listbox(tk.Listbox(
            sub_frame,
            font=(theme.FONT_UI, 12),
            activestyle="none",
            yscrollcommand=sub_scroll.set,
        ))
        sub_scroll.configure(command=self._sub_list.yview)
        self._sub_list.pack(fill="both", expand=True, padx=1, pady=1)
        self._sub_list.bind("<<ListboxSelect>>", self._on_sub_click)

        self._sub_hint = tk.Label(
            sub_frame, text=t("chapter.hint"),
            font=(theme.FONT_UI, 11), bg=theme.SURFACE, fg=TEXT_MUTED,
        )
        self._sub_hint.place(relx=0.5, rely=0.45, anchor="center")

        # --- Barre inférieure : bouton Étudier ---
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=56, pady=28)

        self._study_btn = theme.make_button(
            bottom,
            text=t("chapter.study_btn"),
            kind="primary",
            font=(theme.FONT_UI, 13, "bold"),
            padx=28,
            pady=11,
            state="disabled",
            command=self._do_study,
        )
        self._study_btn.pack(side="right")
        Tooltip(self._study_btn, t("chapter.study_tip"))

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def _on_chapter_click(self, _e):
        sel = self._ch_list.curselection()
        if not sel:
            return

        idx = sel[0]
        chapter = self._level1[idx]
        self._selected = chapter

        self._subs = child_sections_for_chapter(chapter, self._chapters)

        self._sub_list.delete(0, tk.END)
        self._sub_hint.place_forget()

        if self._subs:
            for s in self._subs:
                level = max(2, int(s.get("toc_level", 2) or 2))
                indent = "  " + "    " * max(0, level - 2)
                marker = "↳ " if level >= 3 else ""
                self._sub_list.insert(tk.END, f"{indent}{marker}{s['title']}")
            self._study_btn.configure(state="disabled")  # attendre sélection sous-ch.
        else:
            self._sub_list.insert(tk.END, f"  {t('chapter.whole')}")
            self._study_btn.configure(state="normal")

    def _on_sub_click(self, _e):
        sel = self._sub_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if self._subs:
            self._selected = self._subs[idx]
        self._study_btn.configure(state="normal")

    def refresh_lang(self) -> None:
        chapters = self._chapters
        filename = self._filename
        for child in self.winfo_children():
            child.destroy()
        self._build()
        if filename:
            self.load(filename, chapters)

    def _do_study(self):
        if self._selected:
            self._on_study(self._selected)
