# ui/reading_page.py — Page complète de lecture adaptative
from __future__ import annotations

import logging
import re
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from config.settings import FIGURE_DISPLAY_PAUSE_MS
from db.metacog import ensure_profile
from core.chapter_navigation import normalize_heading_title
from document.postprocess.latex_quality import safe_formula_context_text
from i18n import current_lang, t
from llm.ollama_client import generate_chapter_summary_async, generate_rephrasing_async
from ui import theme
from ui.gauges_panel import GaugesPanel
from ui.inline_reader import InlineReader
from ui.top_nav import TopNav

logger = logging.getLogger("UI.reading_page")

_MIN_QA_LEN = 80   # longueur minimale d'un paragraphe pour le rendu async
_MIN_SECTION_LEN = 80  # longueur minimale d'une section pour déclencher la Q&R
_HEADING_TYPES = {"heading", "subheading", "subsubheading"}
_EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")
_FRONT_MATTER_METADATA_RE = re.compile(
    r"\b(?:doi\s*:|doi\.org|arxiv\s*:|issn\s*:|isbn\s*:|copyright|creative\s+commons)\b",
    re.I,
)
_AFFILIATION_RE = re.compile(
    r"\b(?:university|college|institute|department|school|laborator(?:y|ies)|lab|"
    r"faculty|academy|hospital|centre|center|cnrs|inria|mit|stanford|harvard)\b",
    re.I,
)
_AUTHOR_MARKER_RE = re.compile(r"(?:[*†‡]|\\dagger|\\ddagger|\^\{?\d)")


class ReadingPage(tk.Frame):
    def __init__(
        self,
        master,
        state,
        on_back,
        on_play_pause,
        on_speed_change,
        on_end_session,
        **kwargs,
    ):
        super().__init__(master, bg=theme.BG_ALT, **kwargs)
        self.state = state
        self._on_back = on_back
        self._on_end_session = on_end_session
        self.companion = None
        self.session_mgr = None
        self.doc_title = ""
        self.chapter: dict = {}
        self.llm_available: bool | None = None
        self._active_qa = None
        self._follow_up_qa = None
        self._pending_resume = None
        self._readonly_notice_shown = False
        self._rendered_heading_keys: set[str] = set()
        self._document_type: str = ""
        self._last_slide_page: int = 0
        self._build(on_play_pause, on_speed_change)
        self._tick_timer()

    def _build(self, on_play_pause, on_speed_change) -> None:
        self.top_nav = TopNav(
            self,
            on_back=self._confirm_back,
            on_play_pause=on_play_pause,
            on_speed_change=on_speed_change,
            on_end_session=self.finish_session,
        )
        self.top_nav.pack(side="top", fill="x")

        content = tk.Frame(self, bg=theme.BG_ALT)
        content.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 10))

        self.reader = InlineReader(content, on_paragraph_rephrase=self._on_paragraph_rephrase)
        self.reader.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.gauges = GaugesPanel(content)
        self.gauges.pack(side="right", fill="y")

    def bind_keyboard(self, root) -> None:
        self.top_nav.bind_keyboard(root)

    def set_document(self, filename: str, total_pages: int, engine: str, pdf_path: str | None = None) -> None:
        self.doc_title = filename
        self.reader.set_pdf_path(pdf_path)
        self.top_nav.set_context("", self.state.current_page, total_pages)
        self.top_nav.set_engine(engine)

    def set_document_type(self, doc_type: str) -> None:
        self._document_type = doc_type or ""
        self._last_slide_page = 0

    def set_learning_context(self, companion, session_mgr, doc_title: str, chapter: dict) -> None:
        self.companion = companion
        self.session_mgr = session_mgr
        self.doc_title = doc_title
        self.chapter = chapter
        if session_mgr:
            if session_mgr.subject and session_mgr.subject_level is not None:
                from db.subjects import SUBJECT_LABELS
                label = SUBJECT_LABELS.get(session_mgr.subject, session_mgr.subject.capitalize())
                self.gauges.add_subject_gauge(session_mgr.subject, label, session_mgr.subject_level)
            else:
                self.gauges.remove_subject_gauge()
            self.gauges.update_all(session_mgr.current_gauges())
        else:
            self.gauges.remove_subject_gauge()

    def set_llm_available(self, available: bool) -> None:
        self.llm_available = available
        self.top_nav.set_llm_status("available" if available else "unavailable")

    def start_chapter(self, scope, chapter: dict, doc_title: str) -> None:
        self.reader.clear(hide_loading=False)
        self.doc_title = doc_title
        self.chapter = chapter
        self._active_qa = None
        self._follow_up_qa = None
        self._pending_resume = None
        self._readonly_notice_shown = False
        self._rendered_heading_keys: set[str] = set()
        self._last_slide_page = 0
        self.top_nav.set_context(chapter.get("title", scope.label), chapter["page_start"], self.state.total_pages)

    def clear(self) -> None:
        self.reader.clear()

    def show_pdf_loading(self, message: str | None = None) -> None:
        self.reader.show_loading_overlay(message or t("reading.preparing"))

    def hide_pdf_loading(self) -> None:
        self.reader.hide_loading_overlay()

    def append_char(self, char: str) -> None:
        self.reader.append_char(char)

    def append_block(self, block: dict) -> None:
        if block.get("type") in _HEADING_TYPES:
            key = normalize_heading_title(block.get("text") or "")
            if key in self._rendered_heading_keys:
                return
            self._rendered_heading_keys.add(key)
        page = _block_page_number(block)
        if page:
            self.top_nav.set_context(self.chapter.get("title", ""), int(page), self.state.total_pages)

        # Slides mode: one full-page image per slide + LLM analysis, no individual blocks
        if self._document_type == "slides":
            if page and page != self._last_slide_page:
                self._last_slide_page = page
                self.reader.embed_slide_page(page)
            return

        self.reader.append_block(block)

    def on_paragraph_done(self, block: dict, resume) -> None:
        """Rendu async après streaming — plus de Q&R par paragraphe, la Q&R est à la section."""
        if self._document_type == "slides":
            resume()
            return
        text = (block.get("text") or "").strip()
        if not text:
            resume()
            return
        metadata = block.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        streamed_text_types = {"paragraph", "text", "quote", "abstract"}
        if (
            block.get("type") in streamed_text_types
            and not metadata.get("reader_math_streamed")
            and self.reader.render_completed_paragraph_with_llm(
                block,
                on_complete=lambda _rendered=None: self.after(0, resume),
            )
        ):
            return

        if block.get("type") in streamed_text_types and not metadata.get("reader_math_streamed"):
            self.reader.embed_context_asset(block)

        resume()

    def on_figure_schema(self, block: dict, resume) -> None:
        """Appelé par le moteur pour chaque bloc figure. Stream l'analyse LLM si c'est un schéma,
        puis rappelle resume() pour que la lecture reprenne après la fin du streaming."""
        if not _block_has_schema(block) or self.llm_available is False:
            self.after(FIGURE_DISPLAY_PAUSE_MS, resume)
            return
        self.reader.stream_schema_description(
            block,
            use_llm=True,
            on_done=resume,
        )

    def on_section_complete(
        self,
        section_blocks: list,
        heading_block,
        section_has_latex: bool,
        resume,
    ) -> None:
        """Appelé à la fin de chaque section (titre/sous-titre). Déclenche la Q&R."""
        if self._document_type == "slides":
            resume()
            return
        section_text = _build_section_text(section_blocks)
        heading_label = (heading_block or {}).get("text") or "Section"
        image_paths = _collect_section_image_paths(section_blocks)

        if len(section_text) < _MIN_SECTION_LEN and not image_paths:
            logger.info("Q&R section ignorée: trop courte (%s chars) '%s'", len(section_text), heading_label[:60])
            resume()
            return

        # Construire un bloc synthétique représentant la section entière
        section_block = {
            "type": "paragraph",
            "text": section_text,
            "metadata": {
                "section_image_paths": image_paths,
                "section_has_latex": section_has_latex,
            },
        }
        self.reader.embed_paragraph_rephrase_button(section_block)

        page_start = _first_page(section_blocks, self.state.current_page)
        page_end = _last_page(section_blocks, page_start)

        if self.llm_available is False:
            if not self._readonly_notice_shown:
                self.reader.embed_status(t("reading.ollama_unavailable"))
                self._readonly_notice_shown = True
            resume()
            return

        if self.companion is None:
            logger.info("Q&R section ignorée: compagnon indisponible")
            resume()
            return

        self.set_play_state(False)
        self._pending_resume = resume

        source_id = f"section:{self.state.section_index}:{heading_label[:40]}"
        context = {
            "paragraph": section_text,
            "label": heading_label,
            "source_block_id": source_id,
            "page_start": page_start,
            "page_end": page_end,
            "doc_id": self.state.doc_id,
            "doc_title": self.doc_title,
            "chapter_title": heading_label,
            "chapter_id": self.chapter.get("id"),
            "session_gauges": self.session_mgr.current_gauges() if self.session_mgr else {},
            "recent_question_types": _recent_question_types(self.state.session_history),
            "image_paths": image_paths,
            "scope_type": "section",
        }
        session_id = self.session_mgr.session_id if self.session_mgr else None
        logger.info("Q&R section démarrée '%s' len=%s latex=%s", heading_label[:60], len(section_text), section_has_latex)
        self.companion.start_section_qa(
            context,
            session_id=session_id,
            on_complete=lambda: self.after(0, self._resume_after_qa),
        )

    def on_slide_page_change(self, page: int, resume) -> None:
        self.top_nav.set_context(self.chapter.get("title", ""), page, self.state.total_pages)
        self.reader.embed_slide_page(page, on_analysis_complete=resume)

    def on_math_paragraph_start(self, block: dict, on_done) -> None:  # noqa: D401
        page = _block_page_number(block)
        if page:
            self.top_nav.set_context(self.chapter.get("title", ""), int(page), self.state.total_pages)

        def _complete(rendered: str | None) -> None:
            metadata = block.get("metadata") or {}
            source = ""
            if isinstance(metadata, dict):
                source = str(metadata.get("qa_source_text") or metadata.get("reader_source_text") or "").strip()
            on_done(source or rendered)

        self.reader.stream_math_paragraph(
            block,
            on_complete=_complete,
            use_llm=self.llm_available is not False,
        )

    def on_question_ready(self, question: dict) -> None:
        self.after(0, lambda q=question: (
            self.top_nav.set_llm_status("available"),
            self._show_question(q),
        ))

    def on_answer_evaluated(self, result: dict) -> None:
        self.after(0, lambda r=result: (
            self.top_nav.set_llm_status("available"),
            self._show_evaluation(r),
        ))

    def on_rephrasing_ready(self, rephrasing: dict) -> None:
        self.after(0, lambda r=rephrasing: (
            self.top_nav.set_llm_status("available"),
            self.reader.embed_reformulation(r),
        ))

    def _on_paragraph_rephrase(self, block: dict, on_success, on_error) -> None:
        if self.llm_available is False:
            on_error("LLM indisponible")
            return
        metadata = block.get("metadata") or {}
        image_paths: list[str] = []
        if isinstance(metadata, dict):
            image_paths = list(metadata.get("section_image_paths") or [])
        context = {
            "paragraph": _block_qa_text(block, (block.get("text") or "").strip()),
            "image_paths": image_paths,
            "attempt_count": 0,
        }
        generate_rephrasing_async(context, on_success, on_error)

    def on_paragraph_mask(self, start_char: int, end_char: int, placeholder: str) -> None:
        self.after(0, lambda: self.reader.apply_mask(start_char, end_char, placeholder))

    def on_llm_loading(self, label: str) -> None:
        self.after(0, lambda: self.top_nav.set_llm_status("generating"))
        target = self._active_qa
        if label == "evaluation" and self._qa_is_alive(target):
            self.after(0, lambda block=target: block.show_loading() if self._qa_is_alive(block) else None)
        elif label == "question" and self._qa_is_alive(target):
            self.after(
                0,
                lambda block=target: block.show_pending_question(t("reading.generating_question"))
                if self._qa_is_alive(block)
                else None,
            )

    def on_llm_error(self, message: str) -> None:
        logger.error("Erreur LLM inline : %s", message)
        self.after(0, lambda: (
            self.top_nav.set_llm_status("unavailable"),
            self._show_llm_error(message),
        ))

    def set_play_state(self, is_playing: bool) -> None:
        self.top_nav.set_play_state(is_playing)

    def update_speed_label(self, ms: int) -> None:
        self.top_nav.update_speed_label(ms)

    def set_speed_value(self, ms: int) -> None:
        self.top_nav.set_speed_value(ms)

    def on_chapter_end(self, scope=None) -> None:
        if self.llm_available is False:
            self.reader.embed_status(t("reading.chapter_summary_unavailable"))
            return

        paragraphs_summary = _chapter_paragraphs_summary(scope)
        if _chapter_summary_text_length(paragraphs_summary) < 180:
            logger.warning(
                "Synthèse chapitre ignorée : portée trop courte (%s bloc(s), %s caractères)",
                len(paragraphs_summary),
                _chapter_summary_text_length(paragraphs_summary),
            )
            self.reader.embed_status(t("reading.summary_too_short"))
            return

        self.top_nav.set_llm_status("generating")
        profile = self.session_mgr.profile if self.session_mgr else ensure_profile()
        context = {
            "chapter_title": self.chapter.get("title", ""),
            "paragraphs_summary": paragraphs_summary,
            "metacog_profile": profile,
        }

        def _success(result: dict) -> None:
            self.after(0, lambda r=result: self._show_chapter_summary(r))

        def _error(message: str) -> None:
            logger.error("Synthèse chapitre échouée : %s", message)
            self.after(0, lambda m=message: (
                self.top_nav.set_llm_status("unavailable"),
                self.reader.embed_status(t("reading.chapter_summary_error", msg=m))
            ))

        generate_chapter_summary_async(context, _success, _error)

    def _show_question(self, question: dict) -> None:
        if self._active_qa is not None and not self._qa_is_alive(self._active_qa):
            self._active_qa = None
        if self._active_qa is not None:
            if getattr(self._active_qa, '_mode', None) == 'feedback':
                # Correction visible — ne pas l'écraser, créer un nouveau bloc en dessous
                self._active_qa.clear_pending_status()
                self._active_qa = None
            else:
                self._active_qa.remove_follow_up_form()
                self._active_qa.show_new_question(question)
                return

        def _submit(answer: str, response_time_ms: int) -> None:
            if self.companion:
                self.companion.handle_answer(answer, response_time_ms)

        def _rephrase() -> None:
            if self.companion:
                self.companion.request_new_question()

        self._active_qa = self.reader.embed_qa_block(
            question,
            _submit,
            _rephrase,
            on_reveal_mask=self.reader.reveal_mask,
        )

    def _show_evaluation(self, result: dict) -> None:
        follow_up_answer = result.get("follow_up_answer")
        if follow_up_answer:
            target = self._follow_up_qa or self._active_qa
            if self._qa_is_alive(target):
                target.show_follow_up_answer(follow_up_answer)
            self._follow_up_qa = None
        elif self._qa_is_alive(self._active_qa):
            qa_block = self._active_qa
            completion = result.get("completion", "")
            para_snapshot = self.companion._paragraph_for_llm() if self.companion else ""
            qa_block.show_feedback(
                result.get("verdict", ""),
                result.get("feedback", ""),
                completion,
                result.get("hint", ""),
                on_follow_up=lambda text, block=qa_block, para=para_snapshot: self._on_follow_up(text, block, para),
            )

        if self.session_mgr:
            values = self.session_mgr.update_from_evaluation(
                result,
                response_time_ms=result.get("response_time_ms"),
                consecutive_incorrect=int(result.get("consecutive_incorrect") or 0),
            )
            self.gauges.update_all(values)

        if result.get("verdict") in {"correct", "partial"} and result.get("flashcard"):
            self.reader.embed_flashcard_notif(result["flashcard"])

    def _on_follow_up(self, question_text: str, qa_block=None, paragraph_text: str | None = None) -> None:
        self._follow_up_qa = qa_block or self._active_qa
        if self.companion:
            self.companion.handle_follow_up_question(question_text, paragraph_text)

    def _show_llm_error(self, message: str) -> None:
        self.reader.embed_status(t("reading.llm_error", msg=message))
        self._resume_after_qa()

    def _show_chapter_summary(self, result: dict) -> None:
        self.top_nav.set_llm_status("available")
        self.reader.embed_chapter_summary(result)

    def _resume_after_qa(self) -> None:
        resume = self._pending_resume
        self._pending_resume = None
        self._active_qa = None
        self._follow_up_qa = None
        self.reader.reveal_mask()
        if resume:
            resume()
            self.set_play_state(bool(self.state.is_playing))

    @staticmethod
    def _qa_is_alive(block) -> bool:
        if block is None:
            return False
        try:
            is_alive = getattr(block, "is_alive", None)
            return bool(is_alive()) if callable(is_alive) else bool(block.winfo_exists())
        except tk.TclError:
            return False

    def _confirm_back(self) -> None:
        if messagebox.askyesno(t("reading.confirm_back_title"), t("reading.confirm_back_msg")):
            self._on_back()

    def refresh_lang(self) -> None:
        self.top_nav.refresh_lang()
        self.gauges.refresh_lang()

    def finish_session(self) -> None:
        summary = {}
        if self.session_mgr:
            summary = self.session_mgr.end_session(
                pages_read=max(0, self.state.current_page),
                chapters_completed=[self.chapter.get("title", "")] if self.chapter else [],
            )
        self._on_end_session(summary)

    def _tick_timer(self) -> None:
        if self.session_mgr:
            elapsed = int(time.monotonic() - self.session_mgr.started_monotonic)
            self.top_nav.set_elapsed(elapsed)
        self.after(1000, self._tick_timer)


def _chapter_paragraphs_summary(scope) -> list[dict]:
    if scope is None:
        return []
    blocks = getattr(scope, "blocks", []) or []
    items: list[dict] = []
    for block in blocks:
        btype = block.get("type", "paragraph")
        if btype not in {"heading", "paragraph", "formula", "code", "table"}:
            continue
        text = (block.get("text") or block.get("latex") or block.get("html") or "").strip()
        if not text:
            continue
        items.append({
            "type": btype,
            "page": block.get("page_number"),
            "text": text[:450],
        })
        if len(items) >= 24:
            break
    return items


def _chapter_summary_text_length(items: list[dict]) -> int:
    return sum(
        len(str(item.get("text") or "").strip())
        for item in items
        if item.get("type") != "heading"
    )


def _block_image_paths_for_llm(block: dict) -> list[str]:
    metadata = block.get("metadata") or {}
    paths: list[str] = []

    for asset in metadata.get("llm_assets") or []:
        if isinstance(asset, dict) and asset.get("type") == "image" and asset.get("path"):
            paths.append(str(asset.get("path")))

    for key in ("context_asset_path", "formula_image_path", "table_image_path"):
        path = metadata.get(key) or block.get(key)
        if path:
            paths.append(str(path))

    for path in metadata.get("math_dense_context_assets") or []:
        if path:
            paths.append(str(path))

    for path in metadata.get("reading_unit_image_paths") or []:
        if path:
            paths.append(str(path))

    if block.get("image_path"):
        paths.append(str(block.get("image_path")))

    return _existing_image_paths(paths)[:4]


def _existing_image_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    for raw in dict.fromkeys(str(path) for path in paths if path):
        try:
            if Path(raw).expanduser().exists():
                result.append(raw)
        except OSError:
            continue
    return result


def _block_page_number(block: dict, fallback: int | None = None) -> int | None:
    for key in ("page_number", "page_start", "page"):
        try:
            value = block.get(key)
            if value is not None:
                return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return fallback


def _block_page_end_number(block: dict, fallback: int | None = None) -> int | None:
    for key in ("page_end", "page_number", "page_start", "page"):
        try:
            value = block.get(key)
            if value is not None:
                return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return fallback


def _block_qa_text(block: dict, fallback: str = "") -> str:
    metadata = block.get("metadata") or {}
    if isinstance(metadata, dict):
        source = metadata.get("qa_source_text") or metadata.get("reader_source_text")
        source = source.strip() if isinstance(source, str) else ""
        reading_unit = metadata.get("reading_unit_text")
        if isinstance(reading_unit, str) and reading_unit.strip():
            reading_unit = reading_unit.strip()
            rendered = metadata.get("reader_rendered_text")
            rendered = rendered.strip() if isinstance(rendered, str) else ""
            if source and rendered and reading_unit.startswith(rendered):
                tail = reading_unit[len(rendered):].strip()
                return f"{source}\n\n{tail}".strip() if tail else source
            return reading_unit
        if source:
            return source
    return fallback


def _block_allows_interaction(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("is_metadata", "is_reference", "is_header_footer"):
        if block.get(key) or metadata.get(key):
            return False
    if _looks_like_non_learning_metadata(block):
        _mark_block_metadata(block)
        return False
    if metadata.get("displayable") is False:
        return False
    if metadata.get("interactive") is False:
        return False
    try:
        quality_score = metadata.get("quality_score")
        if quality_score is not None and float(quality_score) < 0.65:
            return False
    except (TypeError, ValueError):
        pass
    return True


def _block_is_metadata(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return bool(block.get("is_metadata") or metadata.get("is_metadata") or _looks_like_non_learning_metadata(block))


def _looks_like_non_learning_metadata(block: dict) -> bool:
    text = _metadata_text(block)
    if not text:
        return False
    if _EMAIL_RE.search(text):
        return True
    if _FRONT_MATTER_METADATA_RE.search(text) and len(text) <= 260:
        return True

    page = _block_page_number(block)
    if page is not None and page > 2:
        return False

    words = re.findall(r"[A-Za-zÀ-ÿ0-9'’.-]+", text)
    if len(words) > 32:
        return False
    if _AUTHOR_MARKER_RE.search(text) and _looks_name_or_affiliation_like(text):
        return True
    if _AFFILIATION_RE.search(text) and _looks_like_affiliation_line(text):
        return True
    return False


def _metadata_text(block: dict) -> str:
    text = block.get("text") or block.get("caption") or ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _looks_name_or_affiliation_like(text: str) -> bool:
    proper_words = re.findall(r"\b[A-Z][A-Za-zÀ-ÿ'’.-]{1,}\b", text)
    return len(proper_words) >= 2 or bool(_AFFILIATION_RE.search(text))


def _looks_like_affiliation_line(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"[.!?;:]\s*$", stripped):
        return False
    lowered = stripped.casefold()
    if re.search(r"\b(?:we|this|these|our|method|model|result|results|figure|table)\b", lowered):
        return False
    return True


def _mark_block_metadata(block: dict) -> None:
    block["is_metadata"] = True
    metadata = block.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        block["metadata"] = metadata
    metadata["is_metadata"] = True
    metadata.setdefault("metadata_reason", "front_matter")


def _block_has_schema(block: dict) -> bool:
    # Only figure-type blocks can carry a schema; formula/context_crop/paragraph cannot.
    if str(block.get("type") or "") != "figure":
        return False
    metadata = block.get("metadata") or {}
    if not isinstance(metadata, dict):
        return False
    # Accept figures tagged as schema OR PDF-cropped (generated on-the-fly during reading).
    has_schema = bool(metadata.get("contains_schema") or metadata.get("pdf_cropped"))
    if not has_schema or not block.get("image_path"):
        return False
    if metadata.get("formula_mode") or metadata.get("contains_inline_math"):
        return False
    reason = str(metadata.get("context_asset_reason") or "").casefold()
    if "math" in reason or "formula" in reason or "crop" in reason:
        return False
    render_mode = str(metadata.get("render_mode") or metadata.get("reader_render_mode") or "").casefold()
    if render_mode in {"pdf_crop", "context_crop_only"}:
        return False
    return bool(_existing_image_paths([str(block.get("image_path"))]))


def _recent_question_types(history: list[dict]) -> list[str]:
    result: list[str] = []
    for item in history or []:
        if isinstance(item, dict) and item.get("question_type"):
            result.append(str(item.get("question_type")))
    return result[-6:]


def _build_section_text(section_blocks: list) -> str:
    """Concatène le texte de tous les blocs d'une section."""
    parts: list[str] = []
    english = current_lang() == "en"
    for block in section_blocks or []:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "paragraph")
        if btype == "formula":
            text = safe_formula_context_text(block.get("latex") or block.get("text"))
            if text:
                parts.append(f"[{'Formula' if english else 'Formule'}] {text}")
            else:
                parts.append("[Displayed formula]" if english else "[Formule affichée]")
        elif btype == "table":
            caption = str(block.get("caption") or "").strip()
            text = str(block.get("markdown") or block.get("text") or "").strip()
            label = "Table" if english else "Tableau"
            if caption and text:
                parts.append(f"[{label}] {caption}\n{text}")
            elif caption:
                parts.append(f"[{label}] {caption}")
            elif text:
                parts.append(f"[{label}]\n{text}")
        elif btype == "figure":
            caption = str(block.get("caption") or block.get("text") or "").strip()
            parts.append(f"[Figure] {caption}" if caption else "[Figure]")
        elif btype == "bullet_list":
            items = block.get("items") or []
            if items:
                parts.append("\n".join(f"• {str(i).strip()}" for i in items if str(i).strip()))
            elif block.get("text"):
                parts.append(str(block.get("text")).strip())
        else:
            text = _block_qa_text(block, (block.get("text") or "").strip())
            if text:
                parts.append(text)
    return "\n\n".join(p for p in parts if p).strip()


def _collect_section_image_paths(section_blocks: list) -> list[str]:
    """Collecte tous les chemins d'images de la section (figures, formules, crops)."""
    paths: list[str] = []
    seen: set[str] = set()
    for block in section_blocks or []:
        if not isinstance(block, dict):
            continue
        for path in _block_image_paths_for_llm(block):
            if path and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths[:8]


def _first_page(section_blocks: list, fallback: int) -> int:
    for block in section_blocks or []:
        p = _block_page_number(block)
        if p:
            return p
    return fallback


def _last_page(section_blocks: list, fallback: int) -> int:
    for block in reversed(section_blocks or []):
        p = _block_page_number(block)
        if p:
            return p
    return fallback
