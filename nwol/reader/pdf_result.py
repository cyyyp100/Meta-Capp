from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PreprocessResult:
    blocks: list[dict[str, Any]]
    score: float
    warnings: list[str]
    stats: dict[str, Any]
