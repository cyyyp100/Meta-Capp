from __future__ import annotations

import re

from document.models import BoundingBox, DocumentBlock


_CONNECTOR_ONLY_RE = re.compile(r"^(?:si|ou encore|avec)$", re.I)
_PROSE_AFTER_MATH_RE = re.compile(
    r"^(?P<formula>\$[^$]{1,48}\$|[A-Za-z0-9_{}^\\+\-*/=<>∼→∞]{1,40})\s+"
    r"(?P<body>(?:Autrement dit|Donc|Ainsi|Comme|En remplaçant|Si\b|On\b).+)$",
    re.I,
)
_PROSE_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{4,}\b")
_MATH_SIGNAL_RE = re.compile(r"(?:\\[A-Za-z]+|[_^{}=<>+\-*/]|[∼→←⇒⇔∞≤≥≠±√α-ωΑ-Ω])")


def repair_display_math_fragments(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Collapse PDF-extracted formula shards into one display formula block.

    Some PDFs render fractions and stacked equations as independent text spans.
    Keeping every shard as a reader block produces unusable reading order
    ("si", numerator, arrow, denominator). The merged block keeps a readable text
    fallback and lets formula_cropper attach the authoritative visual crop.
    """
    split_blocks = _split_leading_math_prefixes(blocks)
    result: list[DocumentBlock] = []
    i = 0
    while i < len(split_blocks):
        block = split_blocks[i]
        if not _is_cluster_start(block):
            result.append(block)
            i += 1
            continue

        group = [block]
        j = i + 1
        while j < len(split_blocks):
            candidate = split_blocks[j]
            if not _is_cluster_piece(candidate):
                if (
                    _is_ignorable_between_formula_fragments(candidate)
                    and j + 1 < len(split_blocks)
                    and _is_cluster_piece(split_blocks[j + 1])
                    and _same_page(group[-1], split_blocks[j + 1])
                    and _close_enough(group[-1], split_blocks[j + 1])
                ):
                    j += 1
                    continue
                break
            if not _same_page(group[-1], candidate):
                break
            if not _close_enough(group[-1], candidate):
                break
            group.append(candidate)
            j += 1

        if _should_merge_group(group):
            result.append(_merge_formula_group(group))
            i = j
            continue

        result.append(block)
        i += 1
    return result


def _split_leading_math_prefixes(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for block in blocks:
        if not _should_split_prefix(result[-1] if result else None, block):
            result.append(block)
            continue

        match = _PROSE_AFTER_MATH_RE.match((block.text or "").strip())
        if match is None:
            result.append(block)
            continue

        formula_text = match.group("formula").strip()
        body_text = match.group("body").strip()
        formula_bbox, body_bbox = _split_bbox_for_prefix(block.bbox, formula_text)
        metadata = dict(block.metadata)
        result.append(
            DocumentBlock(
                type="formula",
                text=_strip_math_dollars(formula_text),
                page=block.page,
                bbox=formula_bbox,
                confidence=min(block.confidence, 0.78),
                metadata={
                    **metadata,
                    "source": "leading_math_prefix_split",
                    "formula_mode": "display",
                    "render_mode": "pdf_crop",
                    "preserve_bbox": True,
                },
            )
        )
        result.append(
            DocumentBlock(
                type="paragraph",
                text=body_text,
                page=block.page,
                bbox=body_bbox,
                confidence=block.confidence,
                metadata={**metadata, "split_leading_math_prefix": True},
            )
        )
    return result


def _should_split_prefix(previous: DocumentBlock | None, block: DocumentBlock) -> bool:
    if previous is None or block.type != "paragraph":
        return False
    if previous.type != "formula":
        return False
    if not _same_page(previous, block) or not _close_enough(previous, block):
        return False
    text = (block.text or "").strip()
    match = _PROSE_AFTER_MATH_RE.match(text)
    if match is None:
        return False
    formula_text = _strip_math_dollars(match.group("formula").strip())
    return bool(_MATH_SIGNAL_RE.search(formula_text) or re.search(r"[uvw]_\{?n\}?", formula_text, re.I))


def _is_cluster_start(block: DocumentBlock) -> bool:
    return block.type == "formula" and _has_display_formula_intent(block)


def _is_cluster_piece(block: DocumentBlock) -> bool:
    if block.type == "formula":
        return True
    if block.type != "paragraph":
        return False
    text = (block.text or "").strip()
    if re.fullmatch(r"\(\d{1,4}\)", text):
        return True
    if _CONNECTOR_ONLY_RE.match(text):
        return True
    if len(text) <= 8 and re.fullmatch(r"[A-Za-z0-9.,]+", text):
        return True
    if len(text) <= 32 and _MATH_SIGNAL_RE.search(text) and _prose_word_count(text) == 0:
        return True
    return False


def _has_display_formula_intent(block: DocumentBlock) -> bool:
    if block.metadata.get("formula_mode") == "display":
        return True
    text = (block.text or block.latex or "").strip()
    return bool(_MATH_SIGNAL_RE.search(text)) and len(_PROSE_WORD_RE.findall(text)) <= 1


def _is_ignorable_between_formula_fragments(block: DocumentBlock) -> bool:
    if block.type != "paragraph":
        return False
    metadata = block.metadata or {}
    if not (metadata.get("semantic_only_block") or metadata.get("displayable") is False):
        return False
    text = re.sub(r"\s+", " ", (block.text or "").strip())
    if not text or len(text) > 180:
        return False
    return bool(_MATH_SIGNAL_RE.search(text) or re.search(r"\(\d{1,4}\)", text))


def _should_merge_group(group: list[DocumentBlock]) -> bool:
    if len(group) < 2:
        return False
    formulas = sum(1 for block in group if block.type == "formula")
    if formulas < 2:
        return False
    bbox = _union_block_bbox(group)
    if bbox is None:
        return True
    if bbox.height > 150.0 or bbox.width > 470.0:
        return False
    prose_words = sum(_prose_word_count(block.text or block.latex or "") for block in group)
    return prose_words <= 2


def _merge_formula_group(group: list[DocumentBlock]) -> DocumentBlock:
    first = group[0]
    bbox = _union_block_bbox(group)
    pieces = [_strip_math_dollars((block.text or block.latex or "").strip()) for block in group]
    text = " ".join(piece for piece in pieces if piece)
    metadata = dict(first.metadata)
    metadata.update(
        {
            "source": "display_math_fragment_merger",
            "formula_mode": "display",
            "render_mode": "pdf_crop",
            "preserve_bbox": True,
            "wide_initial_crop": True,
            "merged_formula_fragments": len(group),
        }
    )
    return DocumentBlock(
        type="formula",
        text=re.sub(r"\s+", " ", text).strip(),
        page=first.page,
        bbox=bbox,
        confidence=min(block.confidence for block in group),
        metadata=metadata,
    )


def _same_page(left: DocumentBlock, right: DocumentBlock) -> bool:
    return left.page is not None and left.page == right.page


def _close_enough(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.bbox is None or right.bbox is None:
        return True
    gap = right.bbox.y0 - left.bbox.y1
    if not (-32.0 <= gap <= 34.0):
        return False
    if _is_equation_number_block(left) or _is_equation_number_block(right):
        return _vertical_overlap(left.bbox, right.bbox) >= min(left.bbox.height, right.bbox.height) * 0.2
    center_delta = abs(left.bbox.center_x - right.bbox.center_x)
    return center_delta <= max(230.0, left.bbox.width * 2.6, right.bbox.width * 2.6)


def _is_equation_number_block(block: DocumentBlock) -> bool:
    return bool(re.fullmatch(r"\(\d{1,4}\)", (block.text or "").strip()))


def _vertical_overlap(left: BoundingBox, right: BoundingBox) -> float:
    return max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))


def _split_bbox_for_prefix(bbox: BoundingBox | None, formula_text: str) -> tuple[BoundingBox | None, BoundingBox | None]:
    if bbox is None:
        return None, None
    prefix_width = min(max(24.0, len(formula_text) * 4.8), bbox.width * 0.42)
    split_x = min(bbox.x1, bbox.x0 + prefix_width)
    formula_bbox = BoundingBox(bbox.x0, bbox.y0, split_x, bbox.y1)
    body_bbox = BoundingBox(min(split_x + 4.0, bbox.x1), bbox.y0, bbox.x1, bbox.y1)
    return formula_bbox, body_bbox


def _union_block_bbox(blocks: list[DocumentBlock]) -> BoundingBox | None:
    bbox = blocks[0].bbox if blocks else None
    for block in blocks[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    return bbox


def _strip_math_dollars(text: str) -> str:
    return text[1:-1].strip() if text.startswith("$") and text.endswith("$") else text


def _prose_word_count(text: str) -> int:
    without_latex = re.sub(r"\\[A-Za-z]+", " ", text)
    return len(_PROSE_WORD_RE.findall(without_latex))
