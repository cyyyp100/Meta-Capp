from __future__ import annotations

import re

from document.models import BoundingBox, DocumentBlock


STOP_TYPES = {"heading", "subheading", "subsubheading", "abstract"}
GROUPABLE_TYPES = {
    "paragraph",
    "text",
    "quote",
    "formula",
    "definition",
    "theorem",
    "example",
    "remark",
    "warning",
    "exercise",
    "question",
}
_MATH_SIGNAL_RE = re.compile(
    r"\\[A-Za-z]+"
    r"|[_^{}=<>]"
    r"|(?<![A-Za-z])\d+\s*/\s*\d+(?![A-Za-z])"
    r"|(?<![A-Za-z])\d+\s*/\s*[A-Za-z](?![A-Za-z])"
    r"|(?<=[A-Za-z0-9⁰¹²³⁴⁵⁶⁷⁸⁹)\]}])\s*[⋅·×]\s*(?=[A-Za-z0-9({\[])"
    r"|[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]"
    r"|[∼~≈→←⇒⇔∞≤≥≠±√∑∫α-ωΑ-Ω]"
    r"|\b(?:ln|log|lim|sin|cos|tan|exp)\b"
)
_PROSE_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{4,}\b")
_WEAK_MATH_TOKENS = frozenset({"_", "^", "{", "}", "(", ")", "[", "]"})
_EQUATION_NUMBER_START_RE = re.compile(r"^\s*(?:\$\s*)?\(\d{1,4}\)")


def group_math_dense_paragraphs_until_heading(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Merge math-dense runs until the next heading-like delimiter.

    OCR math often arrives as several visually related paragraphs/formulas. Trying to
    guess the exact end of the formula is fragile, so once a paragraph is clearly
    math-heavy we keep the whole local explanation together until the next subtitle.
    """
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if not _starts_math_dense_run(block):
            result.append(block)
            i += 1
            continue

        group = [block]
        j = i + 1
        while j < len(blocks):
            candidate = blocks[j]
            if candidate.metadata.get("is_metadata"):
                break
            if candidate.type in STOP_TYPES:
                break
            if candidate.type not in GROUPABLE_TYPES:
                break
            if not _same_page_or_adjacent(group[-1], candidate):
                break
            group.append(candidate)
            j += 1

        if len(group) > 1:
            result.append(_merge_math_dense_group(group))
            i = j
            continue

        result.append(block)
        i += 1
    return result


def _starts_math_dense_run(block: DocumentBlock) -> bool:
    if block.type not in {"paragraph", "text", "quote", "example"}:
        return False
    metadata = block.metadata or {}
    if metadata.get("math_dense_group"):
        return False
    if metadata.get("is_metadata"):
        return False

    text = (block.text or "").strip()
    if _EQUATION_NUMBER_START_RE.match(text):
        return False
    if len(text) < 35:
        return False
    sample = text[:420]
    signals = _meaningful_math_signal_count(sample)
    prose_words = len(_PROSE_WORD_RE.findall(sample))
    if metadata.get("contains_inline_math") or metadata.get("formula_mode") in {"inline", "ambiguous"}:
        return signals >= 5 and (prose_words <= 20 or signals / max(1, prose_words) >= 0.35)
    return (signals >= 5 and (prose_words <= 20 or signals / max(1, prose_words) >= 0.35)) or (
        signals >= 3 and prose_words <= 12
    )


def _meaningful_math_signal_count(text: str) -> int:
    signals = 0
    for token in _MATH_SIGNAL_RE.findall(text):
        if token in _WEAK_MATH_TOKENS:
            continue
        signals += 1
    return signals


def _merge_math_dense_group(blocks: list[DocumentBlock]) -> DocumentBlock:
    first = blocks[0]
    bbox = _union_block_bbox(blocks)
    text = _join_group_text(blocks)
    metadata = dict(first.metadata)
    metadata.update(
        {
            "math_dense_group": True,
            "contains_inline_math": True,
            "formula_mode": "inline",
            "merged_math_dense_blocks": len(blocks),
        }
    )
    image_paths = [
        path
        for block in blocks
        for path in (
            block.image_path,
            (block.metadata or {}).get("formula_image_path"),
            (block.metadata or {}).get("context_asset_path"),
        )
        if path
    ]
    if image_paths:
        metadata["math_dense_context_assets"] = list(dict.fromkeys(str(path) for path in image_paths))

    return DocumentBlock(
        type="paragraph",
        text=text,
        page=first.page,
        bbox=bbox,
        confidence=min(block.confidence for block in blocks),
        metadata=metadata,
    )


def _join_group_text(blocks: list[DocumentBlock]) -> str:
    pieces: list[str] = []
    for block in blocks:
        text = (block.text or block.latex or "").strip()
        if not text:
            continue
        if block.type == "formula" and "$" not in text:
            text = f"${text}$"
        pieces.append(text)
    return re.sub(r"\s+", " ", " ".join(pieces)).strip()


def _same_page_or_adjacent(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.page is None or right.page is None:
        return True
    return abs(int(right.page) - int(left.page)) <= 1


def _union_block_bbox(blocks: list[DocumentBlock]) -> BoundingBox | None:
    bbox = blocks[0].bbox
    for block in blocks[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    return bbox
