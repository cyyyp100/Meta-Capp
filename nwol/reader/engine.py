# reader/engine.py — Moteur de ockageecture progressive robuste
from __future__ import annotations

import logging
import re
from typing import Any, Callable

from config.settings import FIGURE_DISPLAY_PAUSE_MS
from core.scopes import TextScope
from reader.state import ReaderState

logger = logging.getLogger("Reader")


Block = dict[str, Any]

_LATEX_SIGNAL_RE = re.compile(r"\\[A-Za-z]{2,}|\$[^$\n]{1,200}\$")
_MATH_TEXT_SIGNAL_RE = re.compile(
    r"(?<!\w)[A-Za-z]\s*[_^]\s*[A-Za-z0-9{(]"
    r"|[∑∫√∞≈≠≤≥→←↔∈∉∀∃αβγδλμσφψω]"
    r"|\b(?:lim|sin|cos|tan|ln|log|exp)\b\s*[_({]?"
    r"|[A-Za-z0-9]\s*(?:=|≤|≥|≈|≠|<|>)\s*[A-Za-z0-9\\∑∫√∞αβγδλμσφψω]",
)


class ReadingEngine:
    """
    Moteur de lecture progressive.

    Responsabilité volontairement limitée :
    - lire des blocs déjà nettoyés et structurés ;
    - ne jamais planter sur un bloc invalide ;
    - gérer paragraphes, titres, formules, tableaux, code, figures et listes ;
    - éviter les doubles boucles play() et les reprises concurrentes.

    Callbacks attendus :
      - on_char(char: str)
      - on_block(block: dict)
      - on_end()
      - schedule_fn(delay_ms: int, callback: Callable) -> Any
      - on_paragraph_complete(block: dict, resume: Callable) optionnel — rendu async uniquement, pas Q&R
      - on_section_complete(blocks: list, heading: dict|None, has_latex: bool, resume: Callable) optionnel
    """

    INSTANT_BLOCK_TYPES = {
        "formula",
        "figure",
        "code",
        "table",
        "heading",
        "subheading",
        "subsubheading",
        "example",
        "definition",
        "theorem",
        "remark",
        "warning",
    }

    TEXT_BLOCK_TYPES = {"paragraph", "text", "quote", "abstract"}

    HEADING_TYPES = {"heading", "subheading", "subsubheading"}

    def __init__(
        self,
        state: ReaderState,
        on_char: Callable[[str], None],
        on_block: Callable[[Block], None],
        on_end: Callable[[], None],
        schedule_fn: Callable[[int, Callable[[], None]], Any],
        on_paragraph_complete: Callable[[Block, Callable[[], None]], None] | None = None,
        on_math_paragraph: Callable[[Block, Callable[[str | None], None]], None] | None = None,
        on_section_complete: Callable[[list, "Block | None", bool, Callable[[], None]], None] | None = None,
        on_figure_schema: Callable[[Block, Callable[[], None]], None] | None = None,
        # on_prefetch_question gardé pour compatibilité mais ignoré
        on_prefetch_question: Callable[[Block], None] | None = None,
    ):
        self.state = state
        self.on_char = on_char
        self.on_block = on_block
        self.on_end = on_end
        self.on_paragraph_complete = on_paragraph_complete
        self.on_math_paragraph = on_math_paragraph
        self.on_section_complete = on_section_complete
        self.on_figure_schema = on_figure_schema
        self.schedule = schedule_fn
        self.slides_mode: bool = False
        self.on_slide_page_change: Callable[[int, Callable[[], None]], None] | None = None
        self._last_slide_page: int = 0
        self._pending_id: Any | None = None
        self._in_tick = False
        # Section courante
        self._current_section_blocks: list[Block] = []
        self._current_section_heading: Block | None = None
        self._section_has_latex: bool = False

    # ------------------------------------------------------------------
    # Contrôle
    # ------------------------------------------------------------------

    def play(self) -> None:
        if self.state.active_scope is None:
            logger.warning("play() appelé sans scope actif")
            return
        if self.state.qa_active:
            logger.info("Lecture suspendue : Q&R en cours")
            return
        if self.state.is_playing:
            return

        self.state.is_playing = True
        logger.info(
            "Lecture démarrée (mode=char, vitesse=%sms, bloc=%s, char=%s)",
            self.state.speed_ms,
            self.state.current_block_index,
            self.state.char_index,
        )
        self._tick()

    def pause(self) -> None:
        self.state.is_playing = False
        logger.info("Lecture en pause")

    def stop(self) -> None:
        self.state.is_playing = False
        self.state.reset_playback()
        self._reset_section_tracking()

    def set_speed(self, ms: int) -> None:
        self.state.speed_ms = max(0, int(ms))

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._in_tick:
            # Protection contre une schedule_fn qui rappellerait immédiatement.
            logger.debug("Tick réentrant ignoré")
            return

        if not self.state.is_playing:
            return

        self._in_tick = True
        try:
            self._tick_impl()
        except Exception as exc:
            logger.exception("Erreur pendant la lecture : %s", exc)
            # Ne pas bloquer l'application sur un bloc corrompu.
            self._skip_current_block()
            self._schedule_next()
        finally:
            self._in_tick = False

    def _tick_impl(self) -> None:
        scope = self.state.active_scope
        if scope is None:
            self.state.is_playing = False
            return

        blocks = getattr(scope, "blocks", None) or []
        bi = self.state.current_block_index

        if bi >= len(blocks):
            if getattr(scope, "loading_more", False):
                self._schedule_next(max(250, int(self.state.speed_ms)))
                return
            self._finish()
            return

        block = blocks[bi]
        if not isinstance(block, dict):
            logger.warning("Bloc invalide ignoré: %r", block)
            self._skip_current_block()
            self._schedule_next()
            return

        block = self._normalize_runtime_block(block)
        if self._should_skip_runtime_block(block):
            self._skip_current_block()
            self._schedule_next()
            return
        btype = block.get("type", "paragraph")

        # Slides mode: pause on each new page, skip same-page blocks
        if self.slides_mode:
            page = block.get("page_number") or block.get("page") or block.get("page_start")
            if page is not None and self.on_slide_page_change is not None:
                page_int = int(page)
                if page_int != self._last_slide_page:
                    self._last_slide_page = page_int
                    self._pause_for_slide_page_change(page_int)
                    return
            self._skip_current_block()
            self._schedule_next()
            return

        # Frontière de section : heading rencontré après du contenu
        if btype in self.HEADING_TYPES and self._current_section_blocks and self.on_section_complete is not None:
            self._pause_for_section_complete()
            return  # block index inchangé — le heading sera traité après la Q&R

        if btype == "bullet_list":
            self._read_bullet_list(block)
            return

        if btype in self.INSTANT_BLOCK_TYPES:
            self._emit_block(block)
            return

        if btype in self.TEXT_BLOCK_TYPES:
            if self._is_crop_only_text_block(block):
                self._emit_block(block)
                return
            if self._block_has_math(block) and self.on_math_paragraph is not None:
                self._handle_math_paragraph(block)
                return
            self._read_text_block(block)
            return

        # Fallback : si le bloc a du texte, on le lit comme paragraphe.
        if isinstance(block.get("text"), str):
            block["type"] = "paragraph"
            if self._block_has_math(block) and self.on_math_paragraph is not None:
                self._handle_math_paragraph(block)
                return
            self._read_text_block(block)
            return

        logger.warning("Type de bloc non géré ignoré: %s", btype)
        self._skip_current_block()
        self._schedule_next()

    def _normalize_runtime_block(self, block: Block) -> Block:
        btype = block.get("type") or "paragraph"
        block["type"] = str(btype)

        if block["type"] in self.TEXT_BLOCK_TYPES:
            text = block.get("text", "")
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
            block["text"] = text

        return block

    def _should_skip_runtime_block(self, block: Block) -> bool:
        metadata = block.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return bool(
            block.get("is_metadata")
            or metadata.get("is_metadata")
            or block.get("is_reference")
            or metadata.get("is_reference")
            or metadata.get("is_header_footer")
        )

    # ------------------------------------------------------------------
    # Gestion des sections
    # ------------------------------------------------------------------

    def _add_block_to_section(self, block: Block) -> None:
        self._current_section_blocks.append(block)
        self.state.current_section_blocks = list(self._current_section_blocks)
        if not self._section_has_latex and self._block_has_math(block):
            self._section_has_latex = True
            self.state.section_has_latex = True
        if not self._section_has_latex and str(block.get("type") or "") == "formula":
            self._section_has_latex = True
            self.state.section_has_latex = True

    def _start_new_section(self, heading_block: Block) -> None:
        self._current_section_blocks = []
        self._current_section_heading = heading_block
        self._section_has_latex = False
        self.state.current_section_blocks = []
        self.state.current_section_heading = heading_block
        self.state.section_index += 1
        self.state.section_has_latex = False

    def _reset_section_tracking(self) -> None:
        self._current_section_blocks = []
        self._current_section_heading = None
        self._section_has_latex = False

    def _pause_for_section_complete(self) -> None:
        self.state.is_playing = False
        self.state.qa_active = True
        section_blocks = list(self._current_section_blocks)
        heading_block = self._current_section_heading
        section_has_latex = self._section_has_latex
        was_in_tick = self._in_tick

        def _resume() -> None:
            self._current_section_blocks = []
            self._section_has_latex = False
            self.state.current_section_blocks = []
            self.state.section_has_latex = False
            self.state.qa_active = False
            if self.state.active_scope is not None:
                self.play()

        try:
            self.on_section_complete(section_blocks, heading_block, section_has_latex, _resume)
        except Exception as exc:
            logger.exception("Callback section échoué : %s", exc)
            _resume()

        if was_in_tick and self.state.is_playing:
            self._schedule_next()

    def _pause_for_slide_page_change(self, page: int) -> None:
        self.state.is_playing = False
        self.state.qa_active = True
        was_in_tick = self._in_tick

        def _resume() -> None:
            self.state.qa_active = False
            self._skip_current_block()
            if self.state.active_scope is not None:
                self.play()

        try:
            self.on_slide_page_change(page, _resume)
        except Exception as exc:
            logger.exception("Callback slide_page_change échoué : %s", exc)
            _resume()

        if was_in_tick and self.state.is_playing:
            self._schedule_next()

    def _is_crop_only_text_block(self, block: Block) -> bool:
        metadata = block.get("metadata") or {}
        if not isinstance(metadata, dict):
            return False
        if metadata.get("render_mode") != "context_crop_only" and metadata.get("reader_render_mode") != "context_crop_only":
            return False
        return bool(metadata.get("context_asset_path") or block.get("context_asset_path"))

    def _block_has_math(self, block: Block) -> bool:
        metadata = block.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("contains_inline_math") or block.get("contains_inline_math"):
            return True
        formula_mode = metadata.get("formula_mode") or block.get("formula_mode")
        if formula_mode in {"inline", "ambiguous"}:
            return True
        asset_reason = metadata.get("context_asset_reason") or block.get("context_asset_reason")
        if asset_reason == "inline_math":
            return True

        text = block.get("text") or ""
        if not isinstance(text, str) or len(text) < 20:
            return False
        return bool(_LATEX_SIGNAL_RE.search(text) or _MATH_TEXT_SIGNAL_RE.search(text))

    def _handle_math_paragraph(self, block: Block) -> None:
        self.state.is_playing = False

        def _on_math_done(rendered_text: str | None) -> None:
            if isinstance(rendered_text, str) and rendered_text.strip():
                block["text"] = rendered_text.strip()
            metadata = block.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["reader_math_streamed"] = True

            self.state.current_block_index += 1
            self.state.char_index = 0
            self._add_block_to_section(block)

            if self.on_paragraph_complete is not None:
                # Rendu async uniquement (ex. contexte crop), pas Q&R
                def _resume() -> None:
                    self.state.is_playing = True
                    self._schedule_next()
                try:
                    self.on_paragraph_complete(block, _resume)
                except Exception as exc:
                    logger.exception("Callback paragraphe après math échoué : %s", exc)
                    _resume()
                return

            self.state.is_playing = True
            self._schedule_next()

        try:
            self.on_math_paragraph(block, _on_math_done)
        except Exception as exc:
            logger.exception("Rendu paragraphe math échoué, repli lecture texte : %s", exc)
            self.state.is_playing = True
            self._read_text_block(block)

    def _read_text_block(self, block: Block) -> None:
        text = block.get("text", "")
        if not isinstance(text, str) or not text.strip():
            self._skip_current_block()
            self._schedule_next()
            return

        ci = self.state.char_index
        if ci < len(text):
            self.on_char(text[ci])
            self.state.char_index += 1
            self._schedule_next()
            return

        self.on_char("\n")
        self.state.current_block_index += 1
        self.state.char_index = 0
        self._add_block_to_section(block)

        if self.on_paragraph_complete is not None and _block_allows_interaction(block):
            # Rendu async uniquement (ex. contexte crop), pas Q&R
            def _resume() -> None:
                self._schedule_next()
            try:
                self.on_paragraph_complete(block, _resume)
            except Exception as exc:
                logger.exception("Callback paragraphe échoué : %s", exc)
                self._schedule_next()
            return

        self._schedule_next()

    def _read_bullet_list(self, block: Block) -> None:
        items = block.get("items", [])
        if not isinstance(items, list):
            logger.warning("bullet_list invalide ignorée: %r", block)
            self._skip_current_block()
            self._schedule_next()
            return

        cleaned_items = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned_items:
            self._skip_current_block()
            self._schedule_next()
            return

        block = {**block, "items": cleaned_items, "text": "\n".join(f"• {i}" for i in cleaned_items)}
        self._emit_block(block)

    def _emit_block(self, block: Block) -> None:
        self.on_block(block)
        self.state.current_block_index += 1
        self.state.char_index = 0

        btype = str(block.get("type") or "")
        if btype in self.HEADING_TYPES:
            self._start_new_section(block)
        else:
            self._add_block_to_section(block)

        if btype == "figure" and self.on_figure_schema is not None:
            self._pause_for_figure_schema(block)
        else:
            self._schedule_next(self._block_pause_ms(block))

    def _skip_current_block(self) -> None:
        self.state.current_block_index += 1
        self.state.char_index = 0

    def _schedule_next(self, delay_ms: int | None = None) -> None:
        if not self.state.is_playing:
            return
        delay = max(0, int(self.state.speed_ms if delay_ms is None else delay_ms))
        self._pending_id = self.schedule(delay, self._tick)

    def _block_pause_ms(self, block: Block) -> int | None:
        btype = str(block.get("type") or "")

        if btype == "formula":
            metadata = block.get("metadata") or {}
            mode = metadata.get("formula_mode")
            text = str(block.get("latex") or block.get("text") or "")
            try:
                custom_delay = int(block.get("display_pause_ms") or 0)
            except (TypeError, ValueError):
                custom_delay = 0

            if mode == "display":
                formula_delay = min(2800, 800 + len(text) * 20)
            else:
                formula_delay = 500
            return max(int(self.state.speed_ms), formula_delay, custom_delay)

        if btype == "figure":
            try:
                custom_delay = int(block.get("display_pause_ms") or 0)
            except (TypeError, ValueError):
                custom_delay = 0
            return max(int(self.state.speed_ms), int(FIGURE_DISPLAY_PAUSE_MS), custom_delay)

        return None


    def _pause_for_figure_schema(self, block: Block) -> None:
        """Pause le moteur le temps que l'analyse LLM du schéma soit streamée."""
        self.state.is_playing = False
        self.state.qa_active = True
        was_in_tick = self._in_tick

        def _resume() -> None:
            self.state.qa_active = False
            if self.state.active_scope is not None:
                self.play()

        try:
            self.on_figure_schema(block, _resume)
        except Exception as exc:
            logger.exception("Callback figure_schema échoué : %s", exc)
            _resume()

        if was_in_tick and self.state.is_playing:
            self._schedule_next()

    def _finish(self) -> None:
        self.state.is_playing = False
        self.state.char_index = 0
        logger.info("Fin de portée atteinte")

        if self._current_section_blocks and self.on_section_complete is not None:
            def _on_last_section_done() -> None:
                self._current_section_blocks = []
                self._section_has_latex = False
                self.state.current_section_blocks = []
                self.state.section_has_latex = False
                self.on_end()

            try:
                self.on_section_complete(
                    list(self._current_section_blocks),
                    self._current_section_heading,
                    self._section_has_latex,
                    _on_last_section_done,
                )
            except Exception as exc:
                logger.exception("Callback dernière section échoué : %s", exc)
                self.on_end()
        else:
            self.on_end()


def _block_allows_interaction(block: Block) -> bool:
    metadata = block.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("is_metadata", "is_reference", "is_header_footer"):
        if block.get(key) or metadata.get(key):
            return False
    if metadata.get("displayable") is False:
        return False
    if metadata.get("interactive") is False:
        return False
    return True
