from __future__ import annotations

import re


_SPLIT_COMMAND_RE = re.compile(r"\\(?:\s+[A-Za-z]){2,}")
_SPACED_TEXT_CMD_RE = re.compile(r"\\t\s+e\s+x\s+t|\\tex\s+t|\\t\s+h\s+(?:e|ta)", re.I)
_ORPHAN_SCRIPT_RE = re.compile(r"^\s*(?:\${1,2})?\s*[_^]\{")
_EMPTY_SCRIPT_RUN_RE = re.compile(r"(?:\^\{\s*\}\s*\^|_\{\s*\}\s*_|_\{\s*['`]?_\{\s*\}\})")
_BROKEN_COMMAND_WITH_WORD_RE = re.compile(r"\\\s+[A-Za-z]\s+[A-Za-z](?:\s+[A-Za-z])?")
_SINGLE_EMPTY_SCRIPT_RE = re.compile(r"[A-Za-z0-9]\s*[_^]\{\s*\}")
_WORD_FRAGMENT_SUBSCRIPT_RE = re.compile(r"[_^]\{[A-Za-z]{5,}\}")
_NAKED_DOLLAR_MID_FORMULA_RE = re.compile(r"[A-Za-z0-9]\$[A-Za-z]")


def strip_formula_delimiters(text: str | None) -> str:
    value = str(text or "").strip()
    if value.startswith("$$") and value.endswith("$$") and len(value) >= 4:
        return value[2:-2].strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        return value[1:-1].strip()
    return value


def latex_looks_corrupt(text: str | None) -> bool:
    """Detect LaTeX that is very likely OCR/PDF span noise, not a usable formula."""
    value = strip_formula_delimiters(text)
    if not value:
        return False
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return False

    if _SPLIT_COMMAND_RE.search(value) or _SPACED_TEXT_CMD_RE.search(value):
        return True
    if _ORPHAN_SCRIPT_RE.search(value) and re.search(r"\\\s+[A-Za-z]|_\{\s*['`]?_\{", value):
        return True
    if _EMPTY_SCRIPT_RUN_RE.search(value):
        return True
    if _BROKEN_COMMAND_WITH_WORD_RE.search(value) and re.search(r"[_^{}=]", value):
        return True

    opened = value.count("{")
    closed = value.count("}")
    if abs(opened - closed) >= 2:
        return True

    commands = re.findall(r"\\([A-Za-z]+)", value)
    if any(len(command) == 1 for command in commands) and _BROKEN_COMMAND_WITH_WORD_RE.search(value):
        return True

    if _SINGLE_EMPTY_SCRIPT_RE.search(value):
        return True
    if _NAKED_DOLLAR_MID_FORMULA_RE.search(value):
        return True
    if _WORD_FRAGMENT_SUBSCRIPT_RE.search(value) and (
        _SINGLE_EMPTY_SCRIPT_RE.search(value)
        or _NAKED_DOLLAR_MID_FORMULA_RE.search(value)
        or abs(value.count("{") - value.count("}")) >= 1
    ):
        return True

    return False


def safe_formula_context_text(text: str | None) -> str:
    value = strip_formula_delimiters(text)
    if latex_looks_corrupt(value):
        return ""
    return value
