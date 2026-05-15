from __future__ import annotations

import re

from document.models import BoundingBox, DocumentBlock


_MATH_SIGNAL_RE = re.compile(
    r"(?:\\[A-Za-z]+|[_^{}=<>]|[∼~≈→←⇒⇔∞≤≥≠±√∑∫α-ωΑ-Ω⋅·×]|"
    r"(?<![A-Za-z])\d+\s*/\s*\d+(?![A-Za-z])|n!)",
    re.I,
)
_PROSE_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{4,}\b")
_BROKEN_INLINE_RE = re.compile(
    r"(?:\$\s*[A-Za-zÀ-ÿ]{1,2}\$|\bl\s*\$\s*n\b|o\$\s*\(|\$\s*[A-Za-z0-9_{}^\\]+\s*$)",
    re.I,
)
_RESIDUE_PREFIX_RE = re.compile(
    r"^\s*(?:[A-Za-z]|\d+[A-Za-z]?|n!|[A-Za-z]_\{?[A-Za-z0-9]+\}?)\.?\s+"
    r"(?=(?:Pour|Comme|Donc|Ainsi|On|La|Le|Les|En|Il|Elle|This|These|The|We|Given|In)\b)",
    re.I,
)


def cleanup_visual_math_fragments(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Collapse tiny visual formula residues into the nearby crop owner.

    PDF equations are often extracted as a prose line plus several tiny text
    shards. Showing those shards after the crop makes the reader unreadable
    ("n", "n cdot 2", "..."). This pass keeps the visual crop and removes the
    duplicate residue from the text flow.
    """
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if _starts_visual_fragment_group(block):
            group = [block]
            j = i + 1
            while j < len(blocks) and len(group) < 8:
                candidate = blocks[j]
                if not _is_visual_math_residue(candidate):
                    break
                if not _visually_related(group, candidate):
                    break
                group.append(candidate)
                j += 1

            if len(group) > 1:
                result.append(_merge_fragment_group(group))
                i = j
                continue

        cleaned = _strip_leading_residue_after_visual_math(result[-1] if result else None, block)
        result.append(cleaned)
        i += 1

    return result


def _starts_visual_fragment_group(block: DocumentBlock) -> bool:
    if block.type not in {"paragraph", "text", "quote"}:
        return False
    if block.bbox is None or block.page is None:
        return False
    if (block.metadata or {}).get("is_metadata"):
        return False

    metadata = block.metadata or {}
    text = (block.text or "").strip()
    if not text or len(text) > 320:
        return False
    if metadata.get("formula_mode") not in {"inline", "ambiguous"} and not metadata.get("contains_inline_math"):
        return False
    return _has_broken_inline_math(text) or _ends_with_math_residue(text)


def _is_visual_math_residue(block: DocumentBlock) -> bool:
    if block.type not in {"paragraph", "formula", "text"}:
        return False
    if block.bbox is None or block.page is None:
        return False
    if (block.metadata or {}).get("is_metadata"):
        return False

    text = (block.text or block.latex or "").strip()
    if not text or len(text) > 72:
        return False
    if block.type == "formula":
        # Display formulas are standalone blocks, not visual residues
        if (block.metadata or {}).get("formula_mode") == "display":
            return False
        return True
    if len(_PROSE_WORD_RE.findall(text)) > 1:
        return False
    return bool(_MATH_SIGNAL_RE.search(text) or re.fullmatch(r"[A-Za-z0-9.,;:()!\s]+", text))


def _visually_related(group: list[DocumentBlock], candidate: DocumentBlock) -> bool:
    if not group or candidate.bbox is None or candidate.page is None:
        return False
    if any(block.page != candidate.page for block in group if block.page is not None):
        return False
    bbox = _union_block_bbox(group)
    if bbox is None:
        return True
    vertical_gap = candidate.bbox.y0 - bbox.y1
    vertical_overlap = min(bbox.y1, candidate.bbox.y1) - max(bbox.y0, candidate.bbox.y0)
    if vertical_gap > 22.0 or vertical_overlap < -10.0:
        return False
    horizontal_gap = max(candidate.bbox.x0 - bbox.x1, bbox.x0 - candidate.bbox.x1)
    center_delta = abs(candidate.bbox.center_x - bbox.center_x)
    return horizontal_gap <= 42.0 or center_delta <= max(190.0, bbox.width * 1.4, candidate.bbox.width * 2.8)


def _merge_fragment_group(group: list[DocumentBlock]) -> DocumentBlock:
    first = group[0]
    bbox = _union_block_bbox(group)
    text = _join_group_text(group)
    metadata = dict(first.metadata)
    metadata.update(
        {
            "visual_math_fragment_group": True,
            "contains_inline_math": True,
            "formula_mode": "ambiguous",
            "render_mode": "context_crop_only",
            "reader_render_mode": "context_crop_only",
            "merged_visual_math_fragments": len(group),
        }
    )
    return DocumentBlock(
        type="paragraph",
        text=text,
        page=first.page,
        bbox=bbox,
        confidence=min(block.confidence for block in group),
        metadata=metadata,
    )


def _strip_leading_residue_after_visual_math(
    previous: DocumentBlock | None,
    block: DocumentBlock,
) -> DocumentBlock:
    if previous is None or block.type != "paragraph":
        return block
    if previous.bbox is None or block.bbox is None or previous.page != block.page:
        return block
    if not _previous_owns_visual_residue(previous):
        return block
    if block.bbox.y0 - previous.bbox.y1 > 28.0:
        return block

    text = (block.text or "").strip()
    cleaned = _RESIDUE_PREFIX_RE.sub("", text, count=1).strip()
    if cleaned == text or len(cleaned) < 12:
        return block

    return DocumentBlock(
        type=block.type,
        text=cleaned,
        page=block.page,
        bbox=block.bbox,
        level=block.level,
        items=block.items,
        latex=block.latex,
        html=block.html,
        markdown=block.markdown,
        image_path=block.image_path,
        caption=block.caption,
        confidence=block.confidence,
        metadata={**block.metadata, "stripped_leading_formula_residue": True},
        id=block.id,
    )


def _previous_owns_visual_residue(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    return (
        block.type == "formula"
        or metadata.get("visual_math_fragment_group") is True
        or metadata.get("render_mode") == "context_crop_only"
    )


def _has_broken_inline_math(text: str) -> bool:
    return _unescaped_dollar_count(text) % 2 == 1 or bool(_BROKEN_INLINE_RE.search(text))


def _ends_with_math_residue(text: str) -> bool:
    stripped = text.rstrip()
    return bool(re.search(r"(?:\$\s*)?(?:[A-Za-z]|\d+[A-Za-z]?|n!)\s*$", stripped))


def _unescaped_dollar_count(text: str) -> int:
    count = 0
    for index, char in enumerate(text):
        if char == "$" and not _is_escaped(text, index):
            count += 1
    return count


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


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


def _union_block_bbox(blocks: list[DocumentBlock]) -> BoundingBox | None:
    bbox = blocks[0].bbox if blocks else None
    for block in blocks[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    return bbox
