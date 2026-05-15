from __future__ import annotations

import re

from document.models import BoundingBox, DocumentBlock


_PREV_INLINE_END_RE = re.compile(r"(?:\bln|\blog|\bexp|\bsin|\bcos|\btan|[({\[+\-*/=])\s*$")
_NEXT_INLINE_START_RE = re.compile(r"^\s*(?:[).,/+\-*=]|1\s*/|[0-9]+\s*/)")
_INLINE_RESULT_CUE_RE = re.compile(
    r"\b(?:on\s+obtient|on\s+a|donc|ainsi|d'où|d’ou|ce\s+qui\s+donne|il\s+vient)\b",
    re.I,
)
_INLINE_WORD_BRIDGE_RE = re.compile(r"\b(?:by|of|for|to|from|with|and|or|the|la|le|les|de|du|des)\b", re.I)
_MATH_SIGNAL_RE = re.compile(r"(?:\\[A-Za-z]+|[_^=<>+\-*/]|[∼~≈→←⇒⇔∞≤≥≠±√α-ωΑ-Ω])")


def repair_fragmented_inline_formulas(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        ambiguous = _try_merge_ambiguous_inline_sequence(blocks, i)
        if ambiguous is not None:
            merged, next_i = ambiguous
            result.append(merged)
            i = next_i
            continue

        if i + 2 < len(blocks):
            left, middle, right = blocks[i], blocks[i + 1], blocks[i + 2]
            if _looks_like_split_inline_formula(left, middle, right):
                result.append(_merge_as_inline_paragraph(left, middle, right))
                i += 3
                continue
        if i + 1 < len(blocks):
            left, middle = blocks[i], blocks[i + 1]
            if _looks_like_split_inline_formula_pair(left, middle):
                result.append(_merge_as_inline_paragraph(left, middle))
                i += 2
                continue
            if _looks_like_inline_result_formula_pair(left, middle):
                merged = _merge_as_inline_paragraph(left, middle)
                merged.metadata["repaired_inline_result_formula"] = True
                result.append(merged)
                i += 2
                continue

        result.append(blocks[i])
        i += 1
    return result


def _try_merge_ambiguous_inline_sequence(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index + 1 >= len(blocks):
        return None

    first = blocks[index]
    if first.type != "paragraph" or first.metadata.get("formula_mode") not in {"inline", "ambiguous"}:
        return None

    first_text = (first.text or "").strip()
    if not _PREV_INLINE_END_RE.search(first_text):
        return None

    group = [first]
    j = index + 1
    while j < len(blocks) and len(group) < 4:
        candidate = blocks[j]
        if candidate.type != "paragraph" or candidate.metadata.get("formula_mode") not in {"inline", "ambiguous"}:
            break
        if not _same_page(group[-1], candidate):
            break
        if not _blocks_are_reasonably_related(group[-1], candidate):
            break
        candidate_text = (candidate.text or "").strip()
        if not _looks_like_inline_math_tail(candidate_text):
            break
        group.append(candidate)
        j += 1
        if _balanced_inline_expression(" ".join(block.text or "" for block in group)):
            break

    if len(group) < 2:
        return None

    merged = _merge_as_inline_paragraph(*group)
    merged.metadata["repaired_ambiguous_inline_sequence"] = True
    return merged, j


def _looks_like_split_inline_formula(left: DocumentBlock, middle: DocumentBlock, right: DocumentBlock) -> bool:
    if left.type not in {"paragraph", "formula"} or middle.type != "formula" or right.type != "paragraph":
        return False
    if left.type == "formula" and left.metadata.get("formula_mode") != "display":
        return False
    if middle.metadata.get("formula_mode") != "display":
        return False
    if not _same_page(left, middle) or not _same_page(middle, right):
        return False

    formula_text = (middle.text or middle.latex or "").strip()
    if not formula_text or len(formula_text) > 25:
        return False
    if not _blocks_are_close(left, middle) or not _blocks_are_close(middle, right):
        return False

    left_text = (left.text or "").strip()
    right_text = (right.text or "").strip()
    if left.type == "formula" and not _PREV_INLINE_END_RE.search(left_text):
        return False
    return bool(
        _PREV_INLINE_END_RE.search(left_text)
        or _NEXT_INLINE_START_RE.search(right_text)
        or _looks_like_inline_word_bridge(left_text, formula_text, right_text)
    )


def _looks_like_split_inline_formula_pair(left: DocumentBlock, middle: DocumentBlock) -> bool:
    if left.type not in {"paragraph", "formula"} or middle.type != "formula":
        return False
    if left.type == "formula" and left.metadata.get("formula_mode") != "display":
        return False
    if middle.metadata.get("formula_mode") != "display":
        return False
    if not _same_page(left, middle):
        return False
    if not _blocks_are_close(left, middle):
        return False

    left_text = (left.text or "").strip()
    formula_text = (middle.text or middle.latex or "").strip()
    if not formula_text or len(formula_text) > 35:
        return False
    if not _PREV_INLINE_END_RE.search(left_text):
        return False
    return bool(_NEXT_INLINE_START_RE.search(formula_text) or formula_text.count(")") > formula_text.count("("))


def _looks_like_inline_result_formula_pair(left: DocumentBlock, middle: DocumentBlock) -> bool:
    if left.type != "paragraph" or middle.type != "formula":
        return False
    if middle.metadata.get("formula_mode") != "display":
        return False
    if not _same_page(left, middle):
        return False
    if not _blocks_are_close(left, middle):
        return False

    left_text = (left.text or "").strip()
    formula_text = (middle.text or middle.latex or "").strip()
    if not left_text or not formula_text or len(formula_text) > 70:
        return False
    if not (_INLINE_RESULT_CUE_RE.search(left_text) and _MATH_SIGNAL_RE.search(left_text)):
        return False
    if _looks_like_standalone_display_formula(left, middle):
        return False
    return bool(_NEXT_INLINE_START_RE.search(formula_text) or _MATH_SIGNAL_RE.search(formula_text))


def _looks_like_inline_word_bridge(left_text: str, formula_text: str, right_text: str) -> bool:
    if not left_text or not formula_text or not right_text:
        return False
    if len(formula_text) > 60:
        return False
    if left_text.endswith((".", "!", "?", ";", ":")):
        return False
    if not right_text[:1].islower():
        return False
    if not (_MATH_SIGNAL_RE.search(left_text[-40:]) or _MATH_SIGNAL_RE.search(formula_text)):
        return False
    return bool(_MATH_SIGNAL_RE.search(formula_text) and _INLINE_WORD_BRIDGE_RE.search(formula_text))


def _merge_as_inline_paragraph(*blocks: DocumentBlock) -> DocumentBlock:
    left = blocks[0]
    middle = blocks[1]
    pieces = [
        (block.text or block.latex or "").strip()
        for block in blocks
    ]
    text = _join_inline_pieces([piece for piece in pieces if piece])
    bbox = _union_block_bbox(list(blocks))
    metadata = dict(left.metadata)
    metadata.update(
        {
            "contains_inline_math": True,
            "formula_mode": "inline",
            "repaired_fragmented_inline_formula": True,
            "repaired_formula_text": middle.text or middle.latex or "",
        }
    )
    return DocumentBlock(
        type="paragraph",
        text=text,
        page=left.page,
        bbox=bbox,
        confidence=min(block.confidence for block in blocks),
        metadata=metadata,
    )


def _join_inline_pieces(pieces: list[str]) -> str:
    text = ""
    for piece in pieces:
        if not text:
            text = piece
            continue
        if text.endswith(("(", "[", "{", "/", "+", "-", "=")) or piece.startswith((")", "]", "}", "/", "+", "-", "=")):
            text += piece
        else:
            text += " " + piece
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_inline_math_tail(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 45:
        return False
    if re.search(r"\b[A-Za-zÀ-ÿ]{4,}\b", stripped):
        return False
    return bool(re.search(r"[0-9A-Za-z+\-*/=().,\s]+", stripped))


def _balanced_inline_expression(text: str) -> bool:
    return (
        text.count("(") <= text.count(")")
        and text.count("[") <= text.count("]")
        and text.count("{") <= text.count("}")
        and not _PREV_INLINE_END_RE.search(text.strip())
    )


def _same_page(left: DocumentBlock, right: DocumentBlock) -> bool:
    return left.page is not None and left.page == right.page


def _blocks_are_close(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.bbox is None or right.bbox is None:
        return True
    gap = right.bbox.y0 - left.bbox.y1
    return -16 <= gap <= max(28.0, left.bbox.height * 2.5)


def _looks_like_standalone_display_formula(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.bbox is None or right.bbox is None:
        return False
    if right.bbox.width > max(260.0, left.bbox.width * 0.7):
        return True
    if right.bbox.height > max(34.0, left.bbox.height * 2.5):
        return True
    return right.bbox.x0 < left.bbox.x0 - 16.0


def _blocks_are_reasonably_related(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.bbox is None or right.bbox is None:
        return True
    gap = right.bbox.y0 - left.bbox.y1
    if not (-4 <= gap <= max(76.0, left.bbox.height * 6.0)):
        return False
    center_delta = abs(left.bbox.center_x - right.bbox.center_x)
    return center_delta <= max(280.0, left.bbox.width * 1.8, right.bbox.width * 3.0)


def _union_block_bbox(blocks: list[DocumentBlock]) -> BoundingBox | None:
    bbox = blocks[0].bbox
    for block in blocks[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    return bbox
