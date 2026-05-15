# reader/state.py — État courant du lecteur
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.scopes import TextScope
from config.settings import READING_SPEED_INITIAL_MS

logger = logging.getLogger("Reader.state")


@dataclass
class ReaderState:
    # Portée active
    active_scope: TextScope | None = None

    # Lecture progressive
    is_playing: bool = False
    speed_ms: int = READING_SPEED_INITIAL_MS
    char_index: int = 0
    current_block_index: int = 0

    # Document courant
    doc_id: int | None = None
    current_page: int = 1
    total_pages: int = 0

    # Mode chapitre
    chapter_mode: bool = False

    # Extraction / prétraitement
    engine: str = "pymupdf_structured"
    extraction_score: float | None = None
    extraction_warnings: list[str] = field(default_factory=list)

    # Boucle Q&R adaptative
    qa_active: bool = False
    current_question: dict[str, Any] | None = None
    attempt_count: int = 0
    consecutive_incorrect: int = 0
    session_history: list[dict[str, Any]] = field(default_factory=list)

    # Suivi de la section courante
    current_section_blocks: list[dict[str, Any]] = field(default_factory=list)
    current_section_heading: dict[str, Any] | None = None
    section_index: int = 0
    section_has_latex: bool = False

    def reset_playback(self) -> None:
        """Réinitialise uniquement la lecture, pas les métadonnées d'extraction."""
        self.is_playing = False
        self.char_index = 0
        self.current_block_index = 0
        self.qa_active = False
        self.current_question = None
        self.attempt_count = 0
        self.consecutive_incorrect = 0
        self.current_section_blocks = []
        self.current_section_heading = None
        self.section_index = 0
        self.section_has_latex = False
        logger.debug("Playback réinitialisé")

    def set_scope(self, scope: TextScope) -> None:
        self.active_scope = scope
        self.reset_playback()
        logger.info(
            "Portée active : %s '%s' (p.%s–%s, %s blocs)",
            getattr(scope, "scope_type", None),
            getattr(scope, "label", None),
            getattr(scope, "page_start", None),
            getattr(scope, "page_end", None),
            len(getattr(scope, "blocks", []) or []),
        )

    def set_extraction_report(self, engine: str, score: float | None, warnings: list[str] | None = None) -> None:
        self.engine = engine
        self.extraction_score = score
        self.extraction_warnings = warnings or []

    def push_session_history(self, item: dict[str, Any], limit: int = 5) -> None:
        self.session_history.append(item)
        if len(self.session_history) > limit:
            self.session_history = self.session_history[-limit:]
