from __future__ import annotations

import re


_FOMAML_META_GRADIENT_SPLIT_RE = re.compile(
    r"\bg\s*FOMAM\s*\$+\s*L\s*=\s*g\s*\$+\s*k\b"
)
_FOMAML_META_GRADIENT_PLAIN_RE = re.compile(
    r"\bg\s*FOMAM\s*L\s*=\s*g\s*k\b"
)
_BARE_SUBSCRIPT_RE = re.compile(
    r"(?<!\$)"               # pas déjà dans un dollar
    r"(?<!\\)"               # pas échappé
    r"\b([A-Za-z])"          # variable single-lettre
    r"([_^])"                # subscript ou superscript
    r"(\{[^{}$\n]{1,40}\})"  # groupe borné, sans dollar ni newline
    r"(?!\$)"                # pas déjà fermé par dollar
)


def repair_common_inline_math_artifacts(text: str | None) -> str:
    """Repair common LLM/OCR math artifacts before persistence or UI rendering."""
    if not text:
        return ""

    repaired = _repair_split_fomaml_meta_gradient(str(text))
    repaired = _repair_undelimited_fomaml_meta_gradient(repaired)
    repaired = _wrap_bare_subscripts(repaired)
    return repaired


def _repair_split_fomaml_meta_gradient(text: str) -> str:
    return _FOMAML_META_GRADIENT_SPLIT_RE.sub(r"$g^{FOMAML}=g^k$", text)


def _repair_undelimited_fomaml_meta_gradient(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in _math_ranges(text):
        if cursor < start:
            parts.append(_FOMAML_META_GRADIENT_PLAIN_RE.sub(r"$g^{FOMAML}=g^k$", text[cursor:start]))
        parts.append(text[start:end])
        cursor = end

    if cursor < len(text):
        parts.append(_FOMAML_META_GRADIENT_PLAIN_RE.sub(r"$g^{FOMAML}=g^k$", text[cursor:]))

    return "".join(parts) if parts else _FOMAML_META_GRADIENT_PLAIN_RE.sub(r"$g^{FOMAML}=g^k$", text)


def _math_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    mode: str | None = None
    start = 0
    index = 0

    while index < len(text):
        if mode is None:
            if text.startswith(r"\(", index):
                mode = r"\)"
                start = index
                index += 2
                continue
            if text.startswith(r"\[", index):
                mode = r"\]"
                start = index
                index += 2
                continue
            if text.startswith("$$", index) and not _is_escaped(text, index):
                mode = "$$"
                start = index
                index += 2
                continue
            if text[index] == "$" and not _is_escaped(text, index):
                mode = "$"
                start = index
                index += 1
                continue
            index += 1
            continue

        if mode in {r"\)", r"\]"}:
            if text.startswith(mode, index):
                ranges.append((start, index + 2))
                mode = None
                index += 2
                continue
        elif mode == "$$":
            if text.startswith("$$", index) and not _is_escaped(text, index):
                ranges.append((start, index + 2))
                mode = None
                index += 2
                continue
        elif mode == "$" and text[index] == "$" and not _is_escaped(text, index):
            ranges.append((start, index + 1))
            mode = None
            index += 1
            continue

        index += 1

    return ranges


def _wrap_bare_subscripts(text: str) -> str:
    math_ranges = _math_ranges(text)
    parts: list[str] = []
    cursor = 0
    for match in _BARE_SUBSCRIPT_RE.finditer(text):
        if any(start <= match.start() < end for start, end in math_ranges):
            continue
        content = match.group(3)[1:-1]
        if content.isdigit() or "@" in content:
            continue
        parts.append(text[cursor : match.start()])
        parts.append(f"${match.group(1)}{match.group(2)}{match.group(3)}$")
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1
