from __future__ import annotations

import sys
import unicodedata
from pathlib import Path


def ensure_legacy_document_path() -> None:
    root = Path(__file__).resolve().parents[2]
    package_root = root / "nwol"
    legacy = root / "ancien_script"
    package_root_str = str(package_root)
    legacy_str = str(legacy)

    sys.path[:] = [
        path
        for path in sys.path
        if not _same_path(path, package_root_str) and not _same_path(path, legacy_str)
    ]
    if package_root.exists():
        sys.path.insert(0, package_root_str)
    if legacy.exists():
        sys.path.insert(1 if package_root.exists() else 0, legacy_str)


def _same_path(left: str, right: str) -> bool:
    try:
        left_path = str(Path(left).resolve())
        right_path = str(Path(right).resolve())
    except (OSError, RuntimeError):
        left_path = left
        right_path = right
    return _normalize_path(left_path) == _normalize_path(right_path)


def _normalize_path(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()
