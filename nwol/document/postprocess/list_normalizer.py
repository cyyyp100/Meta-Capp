from __future__ import annotations

import re

from document.models import BoundingBox, DocumentBlock


BULLET_RE = re.compile(
    r"^\s*(?:(?:[-*]|\d+[\).]|[a-zA-Z][\).]|[IVXLC]+[\).])\s+|[•▪◦]\s*)(.+)$",
    re.I,
)
INLINE_BULLET_RE = re.compile(r"\s+[•▪◦]\s*")
SECTION_NUMBER_RE = re.compile(r"^\s*\d+\.\s+\d+\.\s+")
SECTION_HEADING_RE = re.compile(
    r"^\s*\d+(?:\.\d+)*\.?\s+"
    r"(?:Introduction|Abstract|Theoretical Background|Literature Review|Methods?|Materials?|"
    r"Results?|Discussion|Conclusions?|References?|Computed Tomography|U-Net|Few-Shot Learning|"
    r"Meta-Learning|Data-Level|Model-Level|Evaluation|Experiments?|Related Work|"
    r"Définition|Definition|Propriété|Proposition|Théorème|Theorem|Exemple|Remarque)\b",
    re.I,
)
UNIT_START_RE = re.compile(
    r"^\s*(?:exemple|définition|definition|remarque|propriété|propriete|théorème|theorem|preuve|démonstration|demonstration)\b",
    re.I,
)


def normalize_lists(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    items: list[str] = []
    start_block: DocumentBlock | None = None
    bbox: BoundingBox | None = None

    def flush() -> None:
        nonlocal items, start_block, bbox
        if not items or start_block is None:
            items = []
            start_block = None
            bbox = None
            return
        cleaned = [item.strip() for item in items if item.strip()]
        if cleaned:
            result.append(
                DocumentBlock(
                    type="bullet_list",
                    text="\n".join(f"• {item}" for item in cleaned),
                    page=start_block.page,
                    bbox=bbox,
                    items=cleaned,
                    confidence=min(1.0, start_block.confidence + 0.05),
                    metadata={"source": "list_normalizer"},
                )
            )
        items = []
        start_block = None
        bbox = None

    previous: DocumentBlock | None = None
    for block in blocks:
        if block.type != "paragraph":
            flush()
            result.append(block)
            previous = block
            continue

        text = block.text.strip()
        if SECTION_NUMBER_RE.match(text) or SECTION_HEADING_RE.match(text):
            flush()
            result.append(block)
            previous = block
            continue

        match = BULLET_RE.match(text)
        if match:
            if start_block is None:
                start_block = block
                bbox = block.bbox
            elif bbox is not None and block.bbox is not None:
                bbox = bbox.union(block.bbox)
            items.extend(_split_inline_bullet_items(match.group(1).strip()))
            previous = block
            continue

        if items and _is_continuation(previous, block):
            parts = _split_inline_bullet_items(text)
            if len(parts) > 1:
                items[-1] = f"{items[-1]} {parts[0]}".strip()
                items.extend(parts[1:])
            else:
                items[-1] = f"{items[-1]} {text}".strip()
            if bbox is not None and block.bbox is not None:
                bbox = bbox.union(block.bbox)
            previous = block
            continue

        flush()
        result.append(block)
        previous = block

    flush()
    return result


def _split_inline_bullet_items(text: str) -> list[str]:
    parts = [part.strip() for part in INLINE_BULLET_RE.split(text or "") if part.strip()]
    return parts or [text.strip()]


def _is_continuation(previous: DocumentBlock | None, block: DocumentBlock) -> bool:
    if previous is None or previous.bbox is None or block.bbox is None:
        return False
    if previous.page != block.page:
        return False
    if UNIT_START_RE.match(block.text.strip()):
        return False
    vertical_gap = block.bbox.y0 - previous.bbox.y1
    return -4.0 <= vertical_gap <= max(18.0, previous.bbox.height * 1.5)
