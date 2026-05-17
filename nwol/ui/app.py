# ui/app.py — Fenêtre principale MetaC-App
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import threading
import time
from pathlib import Path

from core.document import PDFDocument, normalize_chapter_list
from core.companion import AdaptiveCompanion
from core.chapter_navigation import (
    heading_titles_match,
    heading_search_start_page,
    slice_blocks_for_heading_scope,
    slice_blocks_from_heading_to_end,
)
from core.parser import clear_extraction_caches, extract_page_lazy, detect_best_engine, detect_document_type, detect_document_subject, extract_first_pages_text, get_extraction_report
from core.segmentation import segment_blocks
from core.scopes import make_chapter_scope
from document.postprocess.figure_extractor import (
    blocks_have_missing_managed_assets,
    cleanup_all_document_assets,
)
from document.global_index import build_document_global_index
from db.schema import initialize_schema
from db.documents import upsert_document, get_document_by_path
from db.flashcards import get_session_start_cards
from db.metacog import ensure_profile
from db.pages_cache import get_cached_page, get_cached_page_payload, cache_page, clear_cached_pages
from db.document_index import get_document_index, save_document_index
from db.chapters import save_chapters, get_chapters
from db.session_reflections import get_recent_reflection_questions, save_session_reflection
from db.user import DEFAULT_USER_ID, ensure_default_user, get_user_speed, save_user_speed, record_login_and_get_streak, get_user_lang
import i18n as _i18n
from i18n import t
from llm.ollama_client import (
    analyze_meta_cognition_answers_async,
    cancel_pending_generations,
    generate_meta_cognition_questions_async,
    generate_session_summary_async,
    is_ollama_available,
)
from llm.pdf_assistant_queue import get_pdf_llm_queue
from metacog.reflection import fallback_meta_cognition_analysis, normalize_meta_cognition_questions
from metacog.session import SessionManager
from reader.state import ReaderState
from reader.engine import ReadingEngine
from reader.playback import PlaybackController
from db.quiz_questions import get_quiz_questions
from ui.home import HomeScreen
from ui.quiz_page import QuizPage
from ui.chapter_selector import ChapterSelector
from ui.flashcards_page import FlashcardReviewWidget, FlashcardsPage
from ui.metacog_page import MetacogPage
from ui.reading_page import ReadingPage
from ui.session_entry_sas import SessionEntrySas
from ui.session_exit_sas import SessionExitSas
from ui import theme
from config.settings import PDF_READER_INITIAL_PAGES

logger = logging.getLogger("App")


class NWoLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MetaC-App")
        self.geometry("1360x860")
        self.minsize(1060, 680)
        theme.configure_root(self)

        initialize_schema()
        ensure_default_user()
        self._streak = record_login_and_get_streak()
        _i18n.set_lang(get_user_lang(DEFAULT_USER_ID))

        self._doc: PDFDocument | None = None
        self._state = ReaderState(speed_ms=get_user_speed())
        self._engine_name = detect_best_engine()
        self._state.engine = self._engine_name
        self._session_mgr: SessionManager | None = None
        self._companion: AdaptiveCompanion | None = None
        self._pending_chapter: dict | None = None
        self._current_view: str | None = None
        self._reading_generation: int = 0

        self._build_ui()
        _i18n.on_lang_change(self._rebuild_secondary_screens)
        self._bind_engine_and_playback()
        self.reading_page.set_speed_value(self._state.speed_ms)
        self.reading_page.reader.set_llm_speed(self._state.speed_ms)
        self._check_llm_status()
        self._show_home()

        logger.info("Application MetaC-App démarrée")

    # ------------------------------------------------------------------
    # Construction de l'UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu()

        # Conteneur partagé — les trois écrans s'y superposent via place()
        self._container = tk.Frame(self, bg=theme.BG)
        self._container.pack(fill="both", expand=True)

        # Écran 1 : accueil
        self._home_screen = HomeScreen(
            self._container,
            on_import_pdf=self._on_pdf_imported,
            on_flashcards=self._show_flashcards,
            on_profile=self._show_metacog,
            on_quiz=self._show_quiz,
            streak=self._streak,
        )
        self._home_screen.place(relwidth=1, relheight=1)

        # Écran 2 : sélection chapitre
        self._chapter_screen = ChapterSelector(
            self._container,
            on_study=self._on_chapter_chosen,
            on_back=self._show_home,
        )
        self._chapter_screen.place(relwidth=1, relheight=1)

        self._entry_sas = SessionEntrySas(
            self._container,
            on_ready=self._show_start_review,
            on_back=self._back_from_entry_sas,
        )
        self._entry_sas.place(relwidth=1, relheight=1)

        self._start_review_screen = tk.Frame(self._container, bg=theme.BG)
        self._start_review_screen.place(relwidth=1, relheight=1)
        self._start_review_widget = FlashcardReviewWidget(
            self._start_review_screen,
            on_done=self._on_start_review_done,
            title="Révision éclair",
            mode="browse",
        )
        self._start_review_widget.pack(fill="both", expand=True, padx=54, pady=44)

        # Écran 3 : lecteur adaptatif
        self.reading_page = ReadingPage(
            self._container,
            state=self._state,
            on_back=self._back_to_chapter_selector,
            on_play_pause=self._toggle_play,
            on_speed_change=self._on_speed_change,
            on_end_session=self._on_session_end,
        )
        self.reading_page.place(relwidth=1, relheight=1)
        self.reading_page.bind_keyboard(self)

        self._flashcards_page = FlashcardsPage(
            self._container,
            on_back=self._show_home,
        )
        self._flashcards_page.place(relwidth=1, relheight=1)

        self._metacog_page = MetacogPage(
            self._container,
            on_back=self._show_home,
        )
        self._metacog_page.place(relwidth=1, relheight=1)

        self._exit_sas = SessionExitSas(
            self._container,
            on_done=self._on_exit_sas_done,
        )
        self._exit_sas.place(relwidth=1, relheight=1)

        self._quiz_page = QuizPage(
            self._container,
            on_back=self._show_home,
            get_questions=lambda uid, subj=None: get_quiz_questions(uid, subject=subj),
            on_answer=self._on_quiz_answer,
            on_flashcards=self._show_flashcards,
            on_profile=self._show_metacog,
        )
        self._quiz_page.place(relwidth=1, relheight=1)

    def _build_menu(self) -> None:
        self._menubar = tk.Menu(self)
        self._filemenu = tk.Menu(self._menubar, tearoff=0)
        self._filemenu.add_command(label=t("menu.home"),     command=self._show_home)
        self._filemenu.add_command(label=t("menu.open_pdf"), command=self._open_pdf_dialog,
                                   accelerator="Ctrl+O")
        self._filemenu.add_separator()
        self._filemenu.add_command(label=t("menu.quit"),     command=self.on_close)
        self._menubar.add_cascade(label=t("menu.file"), menu=self._filemenu)
        self.bind("<Control-o>", lambda _e: self._open_pdf_dialog())
        self.configure(menu=self._menubar)

    def _rebuild_menu(self) -> None:
        self._filemenu.entryconfigure(0, label=t("menu.home"))
        self._filemenu.entryconfigure(1, label=t("menu.open_pdf"))
        self._filemenu.entryconfigure(3, label=t("menu.quit"))
        self._menubar.entryconfigure(0, label=t("menu.file"))

    def _rebuild_secondary_screens(self) -> None:
        self._rebuild_menu()
        for screen in (
            self._chapter_screen,
            self._entry_sas,
            self.reading_page,
            self._flashcards_page,
            self._exit_sas,
            self._quiz_page,
            self._metacog_page,
        ):
            if hasattr(screen, "refresh_lang"):
                try:
                    screen.refresh_lang()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Navigation entre écrans
    # ------------------------------------------------------------------

    def _show_home(self) -> None:
        self._raise_view("home", self._home_screen)

    def _show_chapter_selector(self) -> None:
        self._raise_view("chapter_selector", self._chapter_screen)

    def _show_entry_sas(self) -> None:
        self._raise_view("entry_sas", self._entry_sas)

    def _show_start_review(self) -> None:
        cards = get_session_start_cards(n=5, doc_id=self._state.doc_id)
        if not cards:
            self._on_start_review_done()
            return
        if self._pending_chapter:
            self._prefetch_chapter_pages(self._pending_chapter)
        self._start_review_widget.load(cards, title=t("app.flash_review"))
        self._raise_view("start_review", self._start_review_screen)

    def _prefetch_chapter_pages(self, chapter: dict) -> None:
        """Pre-extract PDF pages in the background while the user reviews flashcards."""
        if not self._doc or self._state.doc_id is None:
            return
        path = self._doc.path
        doc_id = self._state.doc_id
        engine = self._engine_name
        document_type = self._current_document_type()
        search_page_start = heading_search_start_page(chapter)
        page_end = self._state.total_pages or search_page_start
        initial_end = min(page_end, search_page_start + max(1, int(PDF_READER_INITIAL_PAGES)) - 1)

        def _still_valid() -> bool:
            return self._doc is not None and self._doc.path == path and self._state.doc_id == doc_id

        def _prefetch() -> None:
            prev_tail: list[str] = []
            for p in range(search_page_start, initial_end + 1):
                if not _still_valid():
                    return
                payload = self._get_valid_cached_page_payload_for(doc_id, p, engine)
                if payload is not None and _cached_payload_satisfies_reader_request(payload, validate_with_llm=True):
                    prev_tail = self._tail_block_ids(payload["blocks"])
                    continue
                try:
                    # generation=None so caching always happens even after the view switches to reader
                    blocks, _, _, _, _ = self._extract_and_cache_reader_page(
                        p,
                        enrich_assets=True,
                        prev_page_tail=prev_tail,
                        validate_with_llm=True,
                        pdf_path=path,
                        doc_id=doc_id,
                        engine=engine,
                        document_type=document_type,
                        generation=None,
                    )
                    if blocks:
                        prev_tail = self._tail_block_ids(blocks)
                except Exception as exc:
                    logger.debug("[PREFETCH] page %s: %s", p, exc)
                    return

        logger.info("[PREFETCH] Pré-extraction chapitre lancée: %s pages %s-%s", chapter.get("title"), search_page_start, initial_end)
        threading.Thread(target=_prefetch, daemon=True, name="_prefetch").start()

    def _show_reader(self) -> None:
        self._raise_view("reader", self.reading_page)

    def _show_flashcards(self) -> None:
        self._flashcards_page.load()
        self._raise_view("flashcards", self._flashcards_page)

    def _show_metacog(self) -> None:
        self._metacog_page.load()
        self._raise_view("metacog", self._metacog_page)

    def _show_quiz(self) -> None:
        self._quiz_page.load(user_id=1)
        self._raise_view("quiz", self._quiz_page)

    def _show_exit_sas(self) -> None:
        self._raise_view("exit_sas", self._exit_sas)

    def _raise_view(self, view_name: str, widget: tk.Widget) -> None:
        previous = self._current_view
        self._switch_view(view_name)
        widget.tkraise()
        if previous == view_name or theme.PREFERS_REDUCED_MOTION:
            widget.place_configure(relx=0)
            return
        widget.place_configure(relx=0.018)

        def _update(progress: float) -> None:
            widget.place_configure(relx=0.018 * (1 - theme.ease_out_cubic(progress)))

        def _done() -> None:
            widget.place_configure(relx=0)

        theme.animate(self._container, theme.ANIM_FAST, _update, _done)

    def _switch_view(self, view_name: str) -> None:
        if self._current_view == "reader" and view_name != "reader":
            self._reading_generation += 1
            self._leave_reading_zone()
        elif self._current_view != "reader" and view_name == "reader":
            self._reading_generation += 1
        self._current_view = view_name

    def _leave_reading_zone(self) -> None:
        self.reading_page.hide_pdf_loading()
        self._exit_reading_session("leave_view")

    def _back_to_chapter_selector(self) -> None:
        self._exit_reading_session("back")
        self._show_chapter_selector()

    def _back_from_entry_sas(self) -> None:
        self._pending_chapter = None
        self._show_chapter_selector()

    # ------------------------------------------------------------------
    # Moteur de lecture
    # ------------------------------------------------------------------

    def _bind_engine_and_playback(self) -> None:
        self._reading_engine = ReadingEngine(
            state=self._state,
            on_char=self._on_char,
            on_block=self._on_block,
            on_end=self._on_reading_end,
            schedule_fn=self.after,
            on_paragraph_complete=self.reading_page.on_paragraph_done,
            on_math_paragraph=self.reading_page.on_math_paragraph_start,
            on_section_complete=self.reading_page.on_section_complete,
            on_figure_schema=self.reading_page.on_figure_schema,
        )
        self._playback = PlaybackController(self._reading_engine, self._state)

    def _on_char(self, char: str) -> None:
        self.reading_page.append_char(char)

    def _on_block(self, block: dict) -> None:
        self.reading_page.append_block(block)

    def _on_reading_end(self) -> None:
        self.reading_page.set_play_state(False)
        if self._state.chapter_mode:
            logger.info("Fin de lecture atteinte — synthèse affichée, session conservée ouverte")
            self.reading_page.on_chapter_end(self._state.active_scope)

    # ------------------------------------------------------------------
    # Import PDF
    # ------------------------------------------------------------------

    def _open_pdf_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title=t("app.open_pdf_dialog"),
            filetypes=[(t("home.pdf_files"), "*.pdf"), (t("home.all_files"), "*.*")],
        )
        if path:
            self._on_pdf_imported(path)

    def open_pdf_path(self, path: str) -> None:
        self._on_pdf_imported(path)

    def _on_pdf_imported(self, path: str) -> None:
        """Charge le PDF puis affiche le sas de sélection de chapitre."""
        try:
            self._reading_generation += 1
            try:
                get_pdf_llm_queue().cancel_obsolete()
            except Exception as exc:
                logger.debug("Annulation queue PDF LLM ignorée à l'import: %s", exc)
            clear_extraction_caches()
            if self._doc:
                self._cleanup_all_document_assets()
                self._doc.close()

            self._doc = PDFDocument(path)
            self._doc.open()
            doc_type = detect_document_type(self._doc.path)
            self._engine_name = detect_best_engine(self._doc.path)
            self._state.engine = self._engine_name
            self._state.set_extraction_report(self._engine_name, None, [])
            self._state.total_pages = self._doc.page_count
            self._state.current_page = 1
            self._state.reset_playback()

            doc_info = get_document_by_path(self._doc.path)
            # Fallback keyword immédiat (synchrone, robuste)
            keyword_subject = detect_document_subject(self._doc.path)
            doc_id = upsert_document(
                self._doc.path, self._doc.filename,
                self._doc.page_count, self._engine_name, self._doc.has_toc,
                doc_type=doc_type,
                subject=keyword_subject,
            )
            if keyword_subject:
                from db.subjects import ensure_subject
                ensure_subject(DEFAULT_USER_ID, keyword_subject)
            # Raffinement LLM en arrière-plan (plus précis, gère "suite numérique" etc.)
            self._launch_llm_subject_detection(doc_id, self._doc.path)
            self._state.doc_id = doc_id

            if doc_info and doc_info.get("last_page", 1) > 1:
                self._state.current_page = doc_info["last_page"]

            raw_db_chapters = get_chapters(doc_id)
            db_chapters = normalize_chapter_list(raw_db_chapters, self._doc.page_count)
            if raw_db_chapters and db_chapters != raw_db_chapters:
                save_chapters(doc_id, db_chapters)
            if _looks_like_broken_backmatter_only_toc(db_chapters):
                logger.warning(
                    "TOC DB ignoré: seul le back matter est au niveau 1 pour doc=%s.",
                    doc_id,
                )
                db_chapters = []
            fresh_max_level = max((int(c.get("toc_level", 1) or 1) for c in self._doc.chapters), default=1)
            db_max_level = max((int(c.get("toc_level", 1) or 1) for c in db_chapters), default=1)
            if not db_chapters or fresh_max_level > db_max_level or len(self._doc.chapters) > len(db_chapters):
                save_chapters(doc_id, self._doc.chapters)
                db_chapters = self._doc.chapters

            self.reading_page.set_document(
                self._doc.filename,
                self._doc.page_count,
                self._engine_name,
                self._doc.path,
            )

            self._chapter_screen.load(self._doc.filename, db_chapters)
            self._show_chapter_selector()
            self._launch_document_global_index(self._doc.path, doc_id)
            self._prefetch_start_page(self._doc.path, doc_id, db_chapters)

        except Exception as e:
            logger.error(f"Erreur ouverture PDF : {e}")
            messagebox.showerror(t("app.error.open_pdf_title"), t("app.error.open_pdf_msg", error=e))

    def _launch_document_global_index(self, path: str, doc_id: int) -> None:
        generation = self._reading_generation

        def _index():
            index = None
            try:
                index, status, backend_report = build_document_global_index(doc_id, path)
                save_document_index(
                    doc_id=doc_id,
                    pdf_hash=index.pdf_hash,
                    opendataloader_status=status,
                    detected_document_type=index.document_type,
                    chapters=index.chapters,
                    global_assets={
                        "assets": index.assets,
                        "tables": index.tables,
                        "headings": index.headings,
                    },
                    backend_report=backend_report,
                )
                self.after(0, lambda i=index, s=status, g=generation: self._apply_global_index(path, doc_id, i, s, g))
            except Exception as exc:
                logger.warning("[ODL_INDEX] index global indisponible : %s", exc)

        threading.Thread(target=_index, daemon=True).start()

    def _apply_global_index(self, path: str, doc_id: int, index, status: str, generation: int) -> None:
        if not self._doc or self._doc.path != path or self._state.doc_id != doc_id:
            return
        if status == "complete" and index.chapters:
            merged = normalize_chapter_list([*self._doc.chapters, *index.chapters], self._doc.page_count)
            if merged != self._doc.chapters:
                self._doc.chapters = merged
                save_chapters(doc_id, merged)
                self._chapter_screen.load(self._doc.filename, merged)
        warnings = list(index.warnings or [])
        self._state.set_extraction_report(self._engine_name, index.score, warnings)
        for warning in warnings:
            logger.warning("[ODL_INDEX] %s", warning)

    def _prefetch_start_page(self, path: str, doc_id: int, chapters: list[dict]) -> None:
        generation = self._reading_generation
        first_page = 1
        if chapters:
            try:
                first_page = max(1, int(chapters[0].get("page_start") or 1))
            except (TypeError, ValueError):
                first_page = 1

        def _extract():
            if not self._request_is_current(path, doc_id, generation):
                return
            if get_cached_page(doc_id, first_page, self._engine_name) is not None:
                return
            try:
                result = extract_page_lazy(
                    path,
                    first_page,
                    enrich_assets=False,
                    document_type=self._current_document_type(),
                    validate_with_llm=False,
                    llm_generation=generation,
                )
                if not self._request_is_current(path, doc_id, generation):
                    return
                blocks = segment_blocks(result.to_reader_blocks())
                cache_page(
                    doc_id,
                    first_page,
                    self._engine_name,
                    blocks,
                    page_plan=result.page_plan_dict(),
                    layout_risk=result.layout_risk_dict(),
                    quality_score=result.score,
                    warnings=result.warnings,
                    enrich_assets=False,
                )
                logger.info("[PAGE_EXTRACT] page de départ pré-extraite p.%s", first_page)
            except Exception as exc:
                logger.warning("[PAGE_EXTRACT] pré-extraction page de départ échouée : %s", exc)

        threading.Thread(target=_extract, daemon=True).start()

    def _get_valid_cached_page_payload_for(self, doc_id: int, page: int, engine: str) -> dict | None:
        payload = get_cached_page_payload(doc_id, page, engine)
        blocks = payload["blocks"] if payload else None
        if blocks is not None and blocks_have_missing_managed_assets(blocks):
            logger.info("Cache page invalidé: assets supprimés doc=%s page=%s", doc_id, page)
            clear_extraction_caches()
            return None
        return payload

    def _request_is_current(
        self,
        path: str,
        doc_id: int | None,
        generation: int,
        *,
        view: str | None = None,
    ) -> bool:
        if generation != self._reading_generation:
            return False
        if doc_id is not None and self._state.doc_id != doc_id:
            return False
        if self._doc is None or self._doc.path != path:
            return False
        if view is not None and self._current_view != view:
            return False
        return True

    def _current_document_type(self) -> str | None:
        doc_id = self._state.doc_id
        if doc_id:
            try:
                index = get_document_index(doc_id)
                if index and index.get("detected_document_type"):
                    return str(index["detected_document_type"])
            except Exception as exc:
                logger.debug("Index documentaire indisponible : %s", exc)
        return None

    @staticmethod
    def _block_page(block: dict) -> int:
        for key in ("page_number", "page_start", "page"):
            try:
                value = block.get(key)
                if value is not None:
                    return max(1, int(value))
            except (TypeError, ValueError):
                continue
        return 1

    def _extract_and_cache_reader_page(
        self,
        page: int,
        *,
        enrich_assets: bool = True,
        prev_page_tail: list[str] | None = None,
        validate_with_llm: bool = True,
        pdf_path: str | None = None,
        doc_id: int | None = None,
        engine: str | None = None,
        document_type: str | None = None,
        generation: int | None = None,
    ) -> tuple[list[dict], dict, dict, float | None, list[str]]:
        if not self._doc and not pdf_path:
            return [], {}, {}, None, []
        path = pdf_path or (self._doc.path if self._doc else "")
        cache_doc_id = doc_id if doc_id is not None else self._state.doc_id
        cache_engine = engine or self._engine_name
        if generation is not None and not self._request_is_current(path, cache_doc_id, generation):
            return [], {}, {}, None, []
        result = extract_page_lazy(
            path,
            page,
            prev_page_tail=prev_page_tail,
            enrich_assets=enrich_assets,
            document_type=document_type if document_type is not None else self._current_document_type(),
            validate_with_llm=validate_with_llm,
            llm_generation=None,
        )
        blocks = segment_blocks(result.to_reader_blocks())
        page_plan = result.page_plan_dict()
        layout_risk = result.layout_risk_dict()
        if generation is not None and not self._request_is_current(path, cache_doc_id, generation):
            return [], page_plan, layout_risk, result.score, result.warnings
        if cache_doc_id is not None:
            cache_page(
                cache_doc_id,
                page,
                cache_engine,
                blocks,
                page_plan=page_plan,
                layout_risk=layout_risk,
                quality_score=result.score,
                warnings=result.warnings,
                enrich_assets=enrich_assets,
            )
        return blocks, page_plan, layout_risk, result.score, result.warnings

    @staticmethod
    def _tail_block_ids(blocks: list[dict], limit: int = 4) -> list[str]:
        ids = [str(block.get("id")) for block in blocks if block.get("id")]
        return ids[-limit:]

    # ------------------------------------------------------------------
    # Sélection de chapitre (depuis le sas)
    # ------------------------------------------------------------------

    def _on_chapter_chosen(self, chapter: dict) -> None:
        """Appelé par ChapterSelector après validation."""
        if not self._doc:
            return
        self._pending_chapter = chapter
        self._proceed_to_entry_sas(chapter)

    def _proceed_to_entry_sas(self, chapter: dict) -> None:
        excerpt = self._chapter_excerpt_from_cache(chapter)
        self._entry_sas.load(
            self._doc.filename,
            doc_title=self._doc.filename,
            chapter_title=chapter.get("title", ""),
            profile=ensure_profile(),
            chapter_excerpt=excerpt,
        )
        self._show_entry_sas()
        if not excerpt.strip():
            self._fetch_chapter_excerpt_async(chapter)

    def _fetch_chapter_excerpt_async(self, chapter: dict) -> None:
        if not self._doc or self._state.doc_id is None:
            return
        path = self._doc.path
        doc_id = self._state.doc_id
        generation = self._reading_generation
        engine = self._engine_name
        search_page = heading_search_start_page(chapter)
        total = self._state.total_pages or search_page
        page_end = min(total, max(int(chapter.get("page_start") or 1), search_page) + 1)

        def _extract() -> None:
            for page in range(search_page, page_end + 1):
                if not self._request_is_current(path, doc_id, generation):
                    return
                if get_cached_page(doc_id, page, engine) is None:
                    try:
                        result = extract_page_lazy(
                            path,
                            page,
                            enrich_assets=False,
                            document_type=self._current_document_type(),
                            validate_with_llm=False,
                            llm_generation=generation,
                        )
                        if not self._request_is_current(path, doc_id, generation):
                            return
                        blocks = segment_blocks(result.to_reader_blocks())
                        cache_page(
                            doc_id,
                            page,
                            engine,
                            blocks,
                            page_plan=result.page_plan_dict(),
                            layout_risk=result.layout_risk_dict(),
                            quality_score=result.score,
                            warnings=result.warnings,
                            enrich_assets=False,
                        )
                    except Exception as exc:
                        logger.warning("Extraction différée p.%s échouée: %s", page, exc)
                        return
            if not self._request_is_current(path, doc_id, generation):
                return
            excerpt = self._chapter_excerpt_from_cache(chapter)
            if excerpt.strip():
                self.after(0, lambda e=excerpt: self._entry_sas.update_chapter_excerpt(e))

        threading.Thread(target=_extract, daemon=True).start()

    def _on_start_review_done(self) -> None:
        chapter = self._pending_chapter
        self._pending_chapter = None
        if not chapter:
            self._show_chapter_selector()
            return
        self._show_reader()
        self._study_specific_chapter(chapter)

    def _chapter_excerpt_from_cache(self, chapter: dict) -> str:
        if self._state.doc_id is None:
            return ""
        page_start = int(chapter.get("page_start") or 1)
        search_page_start = heading_search_start_page(chapter)
        page_end = min(self._state.total_pages or page_start, max(page_start, search_page_start) + 1)
        blocks: list[dict] = []
        for page in range(search_page_start, page_end + 1):
            blocks.extend(get_cached_page(self._state.doc_id, page, self._engine_name) or [])
        if blocks:
            blocks = slice_blocks_for_heading_scope(blocks, chapter)
        excerpts: list[str] = []
        for block in blocks:
            text = (block.get("text") or block.get("latex") or block.get("html") or "").strip()
            if text:
                excerpts.append(text)
            if len(" ".join(excerpts)) >= 1200:
                break
        return "\n".join(excerpts)[:1600]

    def _study_specific_chapter(self, chapter: dict) -> None:
        if not self._doc or self._state.doc_id is None:
            return
        self._cancel_chapter_mode()
        self._reading_generation += 1
        self._playback.stop()
        self.reading_page.clear()
        self.reading_page.show_pdf_loading("Préparation des outils de lecture du PDF")

        path = self._doc.path
        doc_id = self._state.doc_id
        engine = self._engine_name
        document_type = self._current_document_type()
        page_start = int(chapter["page_start"])
        search_page_start = heading_search_start_page(chapter)
        page_end = self._state.total_pages
        chapter_scope = {**chapter, "page_start": page_start, "page_end": page_end}
        initial_end = min(page_end, search_page_start + max(1, int(PDF_READER_INITIAL_PAGES)) - 1)

        logger.info(
            "ChapterScope : %s (p. %s–%s, recherche depuis p. %s)",
            chapter["title"],
            page_start,
            page_end,
            search_page_start,
        )
        generation = self._reading_generation

        def _extract():
            all_blocks: list[dict] = []
            extracted_any = False
            prev_tail: list[str] = []

            for p in range(search_page_start, initial_end + 1):
                if not self._request_is_current(path, doc_id, generation, view="reader"):
                    return
                blocks, was_extracted = self._load_chapter_page_blocks(
                    path=path,
                    doc_id=doc_id,
                    engine=engine,
                    document_type=document_type,
                    page=p,
                    prev_tail=prev_tail,
                    generation=generation,
                    validate_with_llm=True,
                )
                extracted_any = extracted_any or was_extracted
                all_blocks.extend(blocks)
                prev_tail = self._tail_block_ids(blocks)

            if extracted_any:
                self._publish_extraction_report()
            scope_blocks = slice_blocks_from_heading_to_end(all_blocks, chapter_scope)
            scope_blocks = self._readable_scope_blocks_or_fallback(scope_blocks, all_blocks, chapter_scope)
            scope_blocks = _ensure_scope_title_block(scope_blocks, chapter_scope)
            scope = make_chapter_scope(chapter_scope["title"], page_start, page_end, scope_blocks)
            scope.loading_more = initial_end < page_end
            self.after(0, lambda s=scope: self._start_chapter_mode_if_current(s, chapter_scope, generation))

            if initial_end >= page_end:
                scope.loading_more = False
                return

            for p in range(initial_end + 1, page_end + 1):
                if not self._request_is_current(path, doc_id, generation, view="reader"):
                    scope.loading_more = False
                    return
                blocks, was_extracted = self._load_chapter_page_blocks(
                    path=path,
                    doc_id=doc_id,
                    engine=engine,
                    document_type=document_type,
                    page=p,
                    prev_tail=prev_tail,
                    generation=generation,
                    validate_with_llm=True,
                )
                extracted_any = extracted_any or was_extracted
                all_blocks.extend(blocks)
                prev_tail = self._tail_block_ids(blocks)
                updated_blocks = slice_blocks_from_heading_to_end(all_blocks, chapter_scope)
                updated_blocks = self._readable_scope_blocks_or_fallback(updated_blocks, all_blocks, chapter_scope)
                updated_blocks = _ensure_scope_title_block(updated_blocks, chapter_scope)
                if len(updated_blocks) > len(scope.blocks):
                    scope.blocks.extend(updated_blocks[len(scope.blocks):])

            scope.loading_more = False
            if extracted_any:
                self._publish_extraction_report()

        threading.Thread(target=_extract, daemon=True).start()

    def _load_chapter_page_blocks(
        self,
        *,
        path: str,
        doc_id: int,
        engine: str,
        document_type: str | None,
        page: int,
        prev_tail: list[str],
        generation: int,
        validate_with_llm: bool,
    ) -> tuple[list[dict], bool]:
        payload = self._get_valid_cached_page_payload_for(doc_id, page, engine)
        if payload is not None and _cached_payload_satisfies_reader_request(
            payload,
            validate_with_llm=validate_with_llm,
        ):
            return payload["blocks"], False
        try:
            blocks, _, _, _, _ = self._extract_and_cache_reader_page(
                page,
                enrich_assets=True,
                prev_page_tail=prev_tail,
                validate_with_llm=validate_with_llm,
                pdf_path=path,
                doc_id=doc_id,
                engine=engine,
                document_type=document_type,
                generation=generation,
            )
            return blocks, True
        except Exception as exc:
            logger.error("Extraction p.%s : %s", page, exc)
            return [], False

    def _readable_scope_blocks_or_fallback(
        self,
        scope_blocks: list[dict],
        all_blocks: list[dict],
        chapter: dict,
    ) -> list[dict]:
        if _has_readable_blocks(scope_blocks):
            return scope_blocks
        page_start = self._block_page(chapter)
        page_end = min(self._state.total_pages or page_start, page_start + 1)
        fallback = [
            block
            for block in all_blocks
            if page_start <= self._block_page(block) <= page_end and _block_is_readable_for_scope(block)
        ]
        if fallback:
            logger.warning(
                "ChapterScope '%s' sans blocs lisibles, fallback page p.%s (%s blocs)",
                chapter.get("title", ""),
                page_start,
                len(fallback),
            )
            return fallback
        return scope_blocks

    def _start_chapter_mode_if_current(self, scope, chapter: dict, generation: int) -> None:
        if self._current_view != "reader" or generation != self._reading_generation:
            return
        self._start_chapter_mode(scope, chapter)

    def _publish_extraction_report(self) -> None:
        if not self._doc:
            return
        try:
            report = get_extraction_report(self._doc.path, self._engine_name)
        except Exception as exc:
            logger.debug("Rapport d'extraction indisponible : %s", exc)
            return
        if report:
            self.after(0, lambda r=report: self._apply_extraction_report(r))

    def _apply_extraction_report(self, report: dict) -> None:
        warnings = list(report.get("warnings") or [])
        self._state.set_extraction_report(
            report.get("engine") or self._engine_name,
            report.get("score"),
            warnings,
        )
        for warning in warnings:
            logger.warning("Extraction : %s", warning)

    # ------------------------------------------------------------------
    # Mode chapitre (depuis la barre de contrôles)
    # ------------------------------------------------------------------

    def _start_chapter_mode(self, scope, chapter: dict) -> None:
        self._state.set_scope(scope)
        self._state.chapter_mode = True
        self._state.current_page = chapter["page_start"]
        from db.documents import get_document_subject
        _subject = get_document_subject(self._state.doc_id) if self._state.doc_id else None
        self._session_mgr = SessionManager(self._state.doc_id, subject=_subject)
        self._companion = AdaptiveCompanion(
            state=self._state,
            on_question=self.reading_page.on_question_ready,
            on_feedback=self.reading_page.on_answer_evaluated,
            on_rephrasing=self.reading_page.on_rephrasing_ready,
            on_mask=self.reading_page.on_paragraph_mask,
            on_loading=self.reading_page.on_llm_loading,
            on_error=self.reading_page.on_llm_error,
        )
        self.reading_page.set_learning_context(
            self._companion,
            self._session_mgr,
            self._doc.filename if self._doc else "",
            chapter,
        )
        self.reading_page.start_chapter(
            scope,
            chapter,
            self._doc.filename if self._doc else "",
        )
        doc_type = self._current_document_type() or ""
        # Lightweight visual detection takes priority for slides (image-heavy PDFs
        # can be mis-classified as "scientific" by the text-keyword pipeline)
        if doc_type != "slides" and self._doc:
            if detect_document_type(self._doc.path) == "slides":
                doc_type = "slides"
        self.reading_page.set_document_type(doc_type)
        self._reading_engine.slides_mode = (doc_type == "slides")
        self._reading_engine._last_slide_page = 0
        self._reading_engine.on_slide_page_change = (
            self.reading_page.on_slide_page_change if doc_type == "slides" else None
        )
        self._playback.play()
        self.reading_page.set_play_state(True)

    def _cancel_chapter_mode(self) -> None:
        if self._state.chapter_mode:
            self._state.chapter_mode = False
        try:
            get_pdf_llm_queue().cancel_obsolete()
        except Exception as exc:
            logger.debug("Annulation queue PDF LLM ignorée: %s", exc)

    # ------------------------------------------------------------------
    # Contrôles lecture
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        state = self._playback.toggle()
        self.reading_page.set_play_state(state == "play")

    def _on_speed_change(self, ms: int) -> None:
        self._playback.set_speed(ms)
        self.reading_page.update_speed_label(self._state.speed_ms)
        self.reading_page.reader.set_llm_speed(self._state.speed_ms)
        save_user_speed(DEFAULT_USER_ID, self._state.speed_ms)

    def _check_llm_status(self) -> None:
        def _check():
            available = is_ollama_available()
            self.after(0, lambda: self.reading_page.set_llm_available(available))
        threading.Thread(target=_check, daemon=True).start()

    def _on_session_end(self, summary: dict) -> None:
        self._exit_reading_session("finish")
        summary = summary if summary.get("duration_s") is not None else {}
        llm_expected = bool(summary and self.reading_page.llm_available is not False)
        self._exit_sas.start_loading(summary, llm_expected=llm_expected)
        self._show_exit_sas()

        if not llm_expected:
            self._exit_sas.set_analysis({})
            self._exit_sas.set_questions([], source="fallback")
            return

        context = {
            "session_data": summary,
            "metacog_profile": summary.get("profile") or {},
        }
        question_context = {
            "session_summary": summary,
            "recent_user_answers": [],
            "previous_end_questions": get_recent_reflection_questions(summary.get("user_id") or 1),
            "user_profile": summary.get("profile") or {},
        }

        def _success(result: dict) -> None:
            self.after(0, lambda r=result: self._exit_sas.set_analysis(r))

        def _error(message: str) -> None:
            logger.error("Synthèse de session échouée : %s", message)
            self.after(0, lambda: self._exit_sas.set_analysis({}))

        def _questions_success(result: dict) -> None:
            questions = normalize_meta_cognition_questions(
                result.get("questions") or [],
                previous_questions=question_context["previous_end_questions"],
                seed_context=summary.get("session_id"),
            )
            self.after(0, lambda q=questions: self._exit_sas.set_questions(q, source="llm"))

        def _questions_error(message: str) -> None:
            logger.error("Questions métacognitives échouées : %s", message)
            self.after(0, lambda: self._exit_sas.set_questions([], source="fallback"))

        generate_session_summary_async(context, _success, _error)
        generate_meta_cognition_questions_async(question_context, _questions_success, _questions_error)

    def _exit_reading_session(self, reason: str) -> None:
        # Idempotent: skip if session is already torn down.
        if not self._state.chapter_mode:
            logger.debug("_exit_reading_session(%s) ignoré: session déjà inactive.", reason)
            return
        logger.info("Sortie de session de lecture: %s", reason)
        self._state.chapter_mode = False
        cancel_pending_generations()
        self._playback.stop()
        self.reading_page.set_play_state(False)
        # End the SessionManager if it was started but not yet closed by reading_page.
        if self._session_mgr is not None and self._session_mgr._ended_summary is None:
            try:
                self._session_mgr.end_session(
                    pages_read=max(0, self._state.current_page),
                    chapters_completed=[],
                )
            except Exception as exc:
                logger.warning("Fin SessionManager échouée dans _exit_reading_session: %s", exc)
        self._cleanup_all_document_assets()

    def _cleanup_all_document_assets(self) -> None:
        try:
            cleanup_all_document_assets()
            clear_extraction_caches()
            if self._state.doc_id is not None:
                clear_cached_pages(self._state.doc_id, self._engine_name)
        except Exception as exc:
            logger.warning("Nettoyage cache assets PDF échoué : %s", exc)

    def _on_exit_sas_done(self, payload: dict) -> None:
        summary = payload.get("summary") or {}
        session_id = summary.get("session_id")
        user_id = summary.get("user_id")
        responses = payload.get("responses") or []
        for response in responses:
            save_session_reflection(
                session_id=session_id,
                user_id=user_id,
                question_text=response.get("question", ""),
                answer_text=response.get("answer", ""),
                question_order=response.get("order", 0),
            )

        if self._session_mgr:
            questions = [response.get("question", "") for response in responses]
            answers = [response.get("answer", "") for response in responses]
            if self.reading_page.llm_available is not False:
                context = {
                    "questions": questions,
                    "answers": answers,
                    "session_context": summary,
                    "user_profile": summary.get("profile") or {},
                }

                def _success(analysis: dict) -> None:
                    self.after(0, lambda a=analysis: self._finalize_exit_sas(a, questions, answers, summary))

                def _error(message: str) -> None:
                    logger.error("Analyse méta-cognitive échouée : %s", message)
                    analysis = fallback_meta_cognition_analysis(questions, answers, summary, summary.get("profile") or {})
                    self.after(0, lambda a=analysis: self._finalize_exit_sas(a, questions, answers, summary))

                analyze_meta_cognition_answers_async(context, _success, _error)
                return

            analysis = fallback_meta_cognition_analysis(questions, answers, summary, summary.get("profile") or {})
            self._finalize_exit_sas(analysis, questions, answers, summary)
            return

        self._finalize_exit_sas(None, [], [], summary)

    def _finalize_exit_sas(
        self,
        meta_analysis: dict | None,
        questions: list[str],
        answers: list[str],
        summary: dict,
    ) -> None:
        if self._session_mgr:
            if meta_analysis:
                self._session_mgr.apply_meta_cognition_analysis(meta_analysis)
            self._session_mgr.finalize_profile()
        self._session_mgr = None
        self._companion = None
        self.reading_page.session_mgr = None
        self.reading_page.companion = None
        self._show_home()

    # ------------------------------------------------------------------
    # Détection de matière via LLM (import PDF)
    # ------------------------------------------------------------------

    def _launch_llm_subject_detection(self, doc_id: int, pdf_path: str) -> None:
        from llm.ollama_client import detect_document_subject_async
        doc_title = Path(pdf_path).stem
        excerpt = extract_first_pages_text(pdf_path, n=2)

        def _on_success(result: dict) -> None:
            subject = result.get("subject")
            if subject:
                self.after(0, lambda s=subject: self._apply_llm_subject(doc_id, s))

        detect_document_subject_async(
            doc_title=doc_title,
            excerpt=excerpt,
            on_success=_on_success,
            on_error=lambda _: None,
        )

    def _apply_llm_subject(self, doc_id: int, subject: str) -> None:
        from db.documents import update_document_subject
        from db.subjects import SUBJECT_LABELS, ensure_subject
        update_document_subject(doc_id, subject)
        ensure_subject(DEFAULT_USER_ID, subject)
        logger.info("Matière LLM appliquée : doc=%s subject=%s", doc_id, subject)

        if self._session_mgr and self._state.doc_id == doc_id and self._session_mgr.subject != subject:
            self._session_mgr.set_subject(subject)
            label = SUBJECT_LABELS.get(subject, subject.capitalize())
            self.reading_page.gauges.add_subject_gauge(subject, label, self._session_mgr.subject_level)

    # ------------------------------------------------------------------
    # Quiz — mise à jour des jauges par matière
    # ------------------------------------------------------------------

    def _on_quiz_answer(self, category: str, correct: bool, details: dict | None = None) -> None:
        from db.session_gauges import record_gauges
        from db.subjects import SUBJECT_LABELS, update_subject_from_answer
        from metacog.profile import update_retention_from_quiz

        details = details or {}
        verdict = details.get("verdict") or ("correct" if correct else "incorrect")
        session_id = self._session_mgr.session_id if self._session_mgr else None
        update_retention_from_quiz(DEFAULT_USER_ID, verdict, session_id=session_id)

        if self._session_mgr and "retention" in self._session_mgr.gauges:
            delta = {"correct": 5.0, "partial": 1.0, "incorrect": -6.0}.get(verdict, 0.0)
            self._session_mgr.gauges["retention"].apply_delta(delta)
            values = self._session_mgr.current_gauges()
            record_gauges(
                self._session_mgr.session_id,
                values,
                t=time.monotonic() - self._session_mgr.started_monotonic,
            )
            self.reading_page.gauges.update_all(values)

        if category in SUBJECT_LABELS:
            new_level = update_subject_from_answer(DEFAULT_USER_ID, category, correct, session_id=session_id)
            if self._session_mgr and self._session_mgr.subject == category:
                values = self._session_mgr.update_subject_level(new_level)
                self.reading_page.gauges.update_all(values)

    # ------------------------------------------------------------------
    # Fermeture
    # ------------------------------------------------------------------

    def on_close(self) -> None:
        self._playback.stop()
        self.reading_page.set_play_state(False)
        self._cleanup_all_document_assets()
        if self._doc:
            self._doc.close()
        from db import close_connection
        close_connection()
        logger.info("Application fermée.")
        self.destroy()


def _looks_like_broken_backmatter_only_toc(chapters: list[dict]) -> bool:
    if len(chapters) < 3:
        return False
    top_level = [
        chapter
        for chapter in chapters
        if _coerce_toc_level(chapter.get("toc_level")) == 1
    ]
    if len(top_level) != 1:
        return False
    return _is_backmatter_title(str(top_level[0].get("title") or ""))


def _coerce_toc_level(value) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _is_backmatter_title(title: str) -> bool:
    normalized = " ".join(title.strip().casefold().split())
    return normalized in {
        "references",
        "bibliography",
        "bibliographie",
        "acknowledgement",
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
        "remerciements",
    }


def _ensure_scope_title_block(blocks: list[dict], chapter: dict) -> list[dict]:
    title = str(chapter.get("title") or "").strip()
    if not title:
        return blocks
    if blocks and _heading_matches_title(blocks[0], title):
        return blocks

    first_visible_heading = next(
        (
            block
            for block in blocks[:4]
            if str(block.get("type") or "") in {"heading", "subheading", "subsubheading"}
        ),
        None,
    )
    if first_visible_heading and _heading_matches_title(first_visible_heading, title):
        return blocks

    try:
        level = min(3, max(1, int(chapter.get("toc_level") or 1)))
    except (TypeError, ValueError):
        level = 1
    try:
        page = max(1, int(chapter.get("page_start") or 1))
    except (TypeError, ValueError):
        page = 1

    synthetic = {
        "type": {1: "heading", 2: "subheading", 3: "subsubheading"}.get(level, "heading"),
        "level": level,
        "text": title,
        "page": page,
        "page_number": page,
        "confidence": 1.0,
        "metadata": {
            "synthetic_scope_title": True,
            "semantic_only_block": True,
        },
    }
    return [synthetic, *blocks]


def _heading_matches_title(block: dict, title: str) -> bool:
    if str(block.get("type") or "") not in {"heading", "subheading", "subsubheading"}:
        return False
    return heading_titles_match(str(block.get("text") or ""), title)


def _cached_payload_satisfies_reader_request(payload: dict, *, validate_with_llm: bool) -> bool:
    if not payload.get("enrich_assets"):
        return False
    if not validate_with_llm:
        return True

    page_plan = payload.get("page_plan")
    layout_risk = payload.get("layout_risk")
    if not isinstance(page_plan, dict) or not isinstance(layout_risk, dict):
        return False
    blocks = payload.get("blocks") or []
    if layout_risk.get("needs_llm_order"):
        has_attempted_order = False
        for block in blocks:
            metadata = block.get("metadata") if isinstance(block, dict) else None
            if isinstance(metadata, dict) and metadata.get("llm_order_status") == "attempted":
                has_attempted_order = True
                break
        if not has_attempted_order:
            return False

    if layout_risk.get("needs_llm_crop"):
        for block in blocks:
            metadata = block.get("metadata") if isinstance(block, dict) else None
            if isinstance(metadata, dict) and metadata.get("llm_crop_status") == "attempted":
                return True
        return False

    return True


def _has_readable_blocks(blocks: list[dict]) -> bool:
    return any(_block_is_readable_for_scope(block) for block in blocks or [])


def _block_is_readable_for_scope(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if (
        block.get("is_metadata")
        or metadata.get("is_metadata")
        or metadata.get("is_reference")
        or metadata.get("is_header_footer")
        or metadata.get("displayable") is False
    ):
        return False
    if block.get("image_path") or metadata.get("context_asset_path") or metadata.get("formula_image_path"):
        return True
    if block.get("items"):
        return True
    return bool((block.get("text") or block.get("latex") or block.get("html") or block.get("markdown") or "").strip())
