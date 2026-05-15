# reader/playback.py — Gestion play/pause et vitesse
from __future__ import annotations

from reader.engine import ReadingEngine
from reader.state import ReaderState
from config.settings import MIN_SPEED_MS, MAX_SPEED_MS


class PlaybackController:
    def __init__(self, engine: ReadingEngine, state: ReaderState):
        self.engine = engine
        self.state = state

    def toggle(self) -> str:
        """Bascule play/pause. Retourne 'play' ou 'pause'."""
        if self.state.qa_active:
            return "pause"
        if self.state.is_playing:
            self.engine.pause()
            return "pause"
        self.engine.play()
        return "play"

    def play(self) -> None:
        self.engine.play()

    def pause(self) -> None:
        self.engine.pause()

    def stop(self) -> None:
        self.engine.stop()

    def set_speed(self, ms: int) -> None:
        ms = max(MIN_SPEED_MS, min(MAX_SPEED_MS, int(ms)))
        self.engine.set_speed(ms)
        self.state.speed_ms = ms

    @property
    def speed_ms(self) -> int:
        return self.state.speed_ms
