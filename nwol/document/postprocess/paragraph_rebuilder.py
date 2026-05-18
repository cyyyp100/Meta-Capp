from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from document.models import BoundingBox, DocumentBlock


# Important : les formules sont protegees. Ne jamais les fusionner dans un paragraphe,
# car les formules PDF complexes sont representees comme des zones 2D, pas comme du texte lineaire fiable.
NON_MERGE_TYPES = {
    "heading",
    "subheading",
    "subsubheading",
    "abstract",
    "definition",
    "theorem",
    "example",
    "remark",
    "warning",
    "exercise",
    "question",
    "formula",
    "bullet_list",
    "figure",
    "table",
    "code",
}

_CONNECTOR_START_RE = re.compile(
    r"^(?:comme|donc|ainsi|or|mais|car|puis|ensuite|alors|on|ce|cette|ces|il|elle|la|le|les|un|une)\b",
    re.I,
)
_STRONG_PARAGRAPH_START_RE = re.compile(
    r"^(?:abstract|résumé|resume|keywords?|mots[- ]clés|mots[- ]cles|exemple|définition|definition|théorème|theorem|proposition|preuve|proof|remarque|remark|corollaire|lemme)\b",
    re.I,
)
_ORPHAN_PREFIX_RE = re.compile(
    r"^\s*(?:[a-zA-Z]|\\[a-zA-Z]+|[_^]\{?[a-zA-Z0-9]+\}?)\s+(?=(?:Comme|Donc|Ainsi|Or|Mais|On|Si)\b)"
)
_LEADING_DISPLAY_EQUATION_RESIDUE_RE = re.compile(
    r"^\s*\(\d{1,4}\)\s+(?:minimi[sz]e|maximi[sz]e|subject\s+to|s\.?t\.?)\s+"
    r"(?:\$[^$]{1,80}\$|[A-Za-z\\_{}^0-9]{1,80})\s+(?=(?:We|This|These|The|Each|Given|In|Here|Thus|Therefore)\b)",
    re.I,
)
_LEADING_EQUATION_NUMBER_RE = re.compile(
    r"^\s*(?:\$\s*)?\(\d{1,4}\)\s+(?=(?:We|This|These|The|Each|Given|In|Here|Thus|Therefore)\b)",
    re.I,
)
_TRAILING_HEADING_PREFIX_RE = re.compile(
    r"^(?P<body>.+?[.!?])\s+(?P<prefix>(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*\.?))$"
)
_MAX_VISUAL_INTERLUDE_BLOCKS = 16


@dataclass(frozen=True, slots=True)
class _PageParagraphProfile:
    line_height: float = 12.0
    normal_gap: float = 3.0

    @property
    def soft_break_gap(self) -> float:
        return max(13.0, self.line_height * 1.45, self.normal_gap * 2.4)

    @property
    def hard_break_gap(self) -> float:
        return max(28.0, self.line_height * 2.7, self.normal_gap * 4.0)


def rebuild_paragraphs(
    blocks: list[DocumentBlock],
    max_chars: int = 900,
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    current: list[DocumentBlock] = []
    profiles = _page_profiles(blocks)
    page_sizes = page_sizes or {}

    def flush() -> None:
        nonlocal current
        if not current:
            return
        block = _merge_paragraph_group(current)
        for chunk in split_long_paragraph(block, max_chars=max_chars):
            result.append(chunk)
        current = []

    previous: DocumentBlock | None = None
    for block in blocks:
        if block.metadata.get("is_caption") or block.metadata.get("is_metadata"):
            flush()
            result.append(block)
            previous = block
            continue

        if block.type in NON_MERGE_TYPES:
            flush()
            result.append(block)
            previous = block
            continue

        if block.type != "paragraph":
            flush()
            result.append(block)
            previous = block
            continue

        if current and _starts_new_paragraph(previous, block, profiles, page_sizes):
            flush()
        current.append(block)
        previous = block

    flush()
    repaired = _repair_split_heading_prefixes(result)
    return _repair_visual_interrupted_paragraphs(repaired, page_sizes=page_sizes, max_chars=max_chars)


def split_long_paragraph(block: DocumentBlock, max_chars: int = 900) -> list[DocumentBlock]:
    text = block.text.strip()
    if len(text) <= max_chars:
        block.text = text
        return [block]

    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences_outside_math(text):
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())

    safe_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            safe_chunks.extend(_split_by_safe_cuts(chunk, max_chars))
        else:
            safe_chunks.append(chunk)
    chunks = safe_chunks

    chunk_bboxes = _split_bbox_for_chunks(block.bbox, chunks)
    return [
        DocumentBlock(
            type=block.type,
            text=chunk,
            page=block.page,
            bbox=chunk_bboxes[index],
            confidence=block.confidence,
            metadata=dict(block.metadata),
        )
        for index, chunk in enumerate(chunks)
        if chunk
    ]


def _split_bbox_for_chunks(bbox: BoundingBox | None, chunks: list[str]) -> list[BoundingBox | None]:
    if bbox is None or len(chunks) <= 1:
        return [bbox for _ in chunks]

    weights = [max(1, len(chunk)) for chunk in chunks]
    total = float(sum(weights))
    y = bbox.y0
    result: list[BoundingBox | None] = []
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            y1 = bbox.y1
        else:
            y1 = y + bbox.height * (weight / total)
        result.append(BoundingBox(bbox.x0, y, bbox.x1, max(y, y1)))
        y = y1
    return result


def _split_sentences_outside_math(text: str) -> list[str]:
    ranges = _math_ranges(text)
    sentences: list[str] = []
    start = 0
    index = 0

    while index < len(text):
        if text[index] in ".!?;:" and not _index_in_ranges(index, ranges):
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor > index + 1:
                sentence = text[start:cursor].strip()
                if sentence:
                    sentences.append(sentence)
                start = cursor
                index = cursor
                continue
        index += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def _split_by_safe_cuts(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = _find_safe_cut_outside_math(remaining, max_chars)
        if cut == -1:
            cut = min(len(remaining), max_chars) - 1
        chunks.append(remaining[: cut + 1].strip())
        remaining = remaining[cut + 1 :].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _find_safe_cut_outside_math(text: str, max_chars: int) -> int:
    if not text:
        return -1

    limit = min(len(text), max(1, int(max_chars)))
    ranges = _math_ranges(text)
    for start, end in ranges:
        if start < limit < end:
            limit = min(len(text), end)
            break

    for index in range(limit - 1, 0, -1):
        if text[index] in ".!?;:" and not _index_in_ranges(index, ranges):
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if not next_char or next_char.isspace():
                return index

    for index in range(limit - 1, 0, -1):
        if text[index].isspace() and not _index_in_ranges(index, ranges):
            return index

    for start, end in ranges:
        if start < max_chars < end:
            return end - 1
    return -1


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

    if mode is not None:
        ranges.append((start, len(text)))
    return ranges


def _index_in_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _merge_paragraph_group(blocks: list[DocumentBlock]) -> DocumentBlock:
    first = blocks[0]
    bbox: BoundingBox | None = first.bbox
    texts: list[str] = []
    pages = [int(block.page) for block in blocks if block.page is not None]
    merged_across_columns = False
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if texts and texts[-1].endswith("-"):
            texts[-1] = _join_hyphenated_line(texts[-1], text)
        else:
            texts.append(text)
        if (
            bbox is not None
            and block.bbox is not None
            and block.page == first.page
            and _bbox_is_column_flow_continuation(bbox, block.bbox)
        ):
            merged_across_columns = True
        elif bbox is not None and block.bbox is not None and block.page == first.page:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox
    metadata = dict(first.metadata)
    metadata["merged_lines"] = len(blocks)
    if merged_across_columns:
        metadata["merged_across_columns"] = True
    if pages and max(pages) > min(pages):
        metadata["page_start"] = min(pages)
        metadata["page_end"] = max(pages)
        metadata["merged_across_pages"] = True
    text = _clean_merged_text(" ".join(texts).strip())
    return DocumentBlock(
        type="paragraph",
        text=text,
        page=first.page,
        bbox=bbox,
        confidence=min(block.confidence for block in blocks),
        metadata=metadata,
    )


def _repair_visual_interrupted_paragraphs(
    blocks: list[DocumentBlock],
    *,
    page_sizes: dict[int, tuple[float, float]],
    max_chars: int,
) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        left = blocks[i]
        if left.type != "paragraph":
            result.append(left)
            i += 1
            continue

        interlude: list[DocumentBlock] = []
        j = i + 1
        while (
            j < len(blocks)
            and len(interlude) < _MAX_VISUAL_INTERLUDE_BLOCKS
            and _is_visual_interlude_block(blocks[j])
        ):
            interlude.append(blocks[j])
            j += 1

        if (
            interlude
            and j < len(blocks)
            and _can_merge_around_visual_interlude(left, blocks[j], interlude, page_sizes)
        ):
            merged = _merge_interrupted_paragraph(left, blocks[j])
            result.extend(split_long_paragraph(merged, max_chars=max_chars))
            result.extend(interlude)
            i = j + 1
            continue

        result.append(left)
        i += 1

    return result


def _is_visual_interlude_block(block: DocumentBlock) -> bool:
    if block.type in {"figure", "table"}:
        return True
    if block.type != "paragraph":
        return False
    metadata = block.metadata or {}
    return bool(metadata.get("is_caption") or metadata.get("caption_isolated"))


def _can_merge_around_visual_interlude(
    left: DocumentBlock,
    right: DocumentBlock,
    interlude: list[DocumentBlock],
    page_sizes: dict[int, tuple[float, float]],
) -> bool:
    if right.type != "paragraph":
        return False
    if left.metadata.get("is_caption") or right.metadata.get("is_caption"):
        return False
    left_text = (left.text or "").rstrip()
    right_text = (right.text or "").strip()
    if len(left_text) < 40 or len(right_text) < 30:
        return False
    if _STRONG_PARAGRAPH_START_RE.match(right_text):
        return False
    if left_text.endswith((".", "!", "?", "”", '"', "'")):
        return False
    if not _has_cross_page_continuation_signal(left_text, right_text):
        return False

    try:
        left_page = int(left.page or 0)
        right_page = int(right.page or 0)
    except (TypeError, ValueError):
        return False
    if left_page <= 0 or right_page < left_page or right_page > left_page + 1:
        return False

    if right_page == left_page:
        return True

    if not page_sizes:
        return True
    page_height = float((page_sizes.get(left_page) or (0.0, 0.0))[1] or 0.0)
    if page_height <= 0.0 or left.bbox is None:
        return True
    if left.bbox.y1 < page_height * 0.72:
        return False
    current_height = float((page_sizes.get(right_page) or (0.0, 0.0))[1] or 0.0)
    has_table_interlude = any(block.type == "table" for block in interlude)
    if (
        not has_table_interlude
        and current_height > 0.0
        and right.bbox is not None
        and right.bbox.y0 > current_height * 0.30
    ):
        return False
    return any(int(block.page or 0) == right_page for block in interlude)


def _merge_interrupted_paragraph(left: DocumentBlock, right: DocumentBlock) -> DocumentBlock:
    text = _clean_merged_text(f"{(left.text or '').rstrip()} {(right.text or '').strip()}".strip())
    metadata = dict(left.metadata)
    metadata["merged_across_visual_interlude"] = True
    if left.page is not None and right.page is not None and int(left.page) != int(right.page):
        metadata["page_start"] = min(int(left.page), int(right.page))
        metadata["page_end"] = max(int(left.page), int(right.page))
        metadata["merged_across_pages"] = True
    source_ids = _source_ids(left) + [item for item in _source_ids(right) if item not in _source_ids(left)]
    if source_ids:
        metadata["source_blocks"] = source_ids
    return DocumentBlock(
        type="paragraph",
        text=text,
        page=left.page,
        bbox=left.bbox,
        confidence=min(left.confidence, right.confidence),
        metadata=metadata,
    )


def _source_ids(block: DocumentBlock) -> list[str]:
    ids = [str(item) for item in (block.metadata or {}).get("source_blocks") or [] if item]
    if block.id and block.id not in ids:
        ids.append(block.id)
    return ids


def _bbox_is_column_flow_continuation(current_bbox: BoundingBox, next_bbox: BoundingBox) -> bool:
    vertical_regression = current_bbox.y0 - next_bbox.y0
    if vertical_regression <= 24.0:
        return False
    horizontal_shift = abs(next_bbox.center_x - current_bbox.center_x)
    return horizontal_shift > max(90.0, min(current_bbox.width, next_bbox.width) * 0.45)


def _starts_new_paragraph(
    previous: DocumentBlock | None,
    block: DocumentBlock,
    profiles: dict[int, _PageParagraphProfile],
    page_sizes: dict[int, tuple[float, float]],
) -> bool:
    if previous is None or previous.bbox is None or block.bbox is None:
        return False
    if previous.page != block.page:
        return not _looks_like_cross_page_continuation(previous, block, page_sizes)
    if previous.type != "paragraph":
        return True

    profile = profiles.get(int(block.page or 0), _PageParagraphProfile())
    vertical_gap = block.bbox.y0 - previous.bbox.y1
    current_text = block.text.strip()
    previous_text = previous.text.rstrip()

    if _looks_like_display_math_fragment(block):
        return True

    if _looks_like_column_restart(previous, block, profile):
        vertical_regression = previous.bbox.y0 - block.bbox.y0
        is_deep_column_jump = vertical_regression > max(50.0, profile.line_height * 4.0)
        if _looks_like_reverse_column_jump(previous, block, profile):
            return True
        if previous_text.endswith("-") and _looks_like_column_flow_continuation(previous_text, current_text):
            return False
        if not is_deep_column_jump and _looks_like_column_flow_continuation(previous_text, current_text):
            return False
        return True

    if _looks_like_parallel_column_line(previous, block, profile):
        return True

    if _looks_like_adjacent_cross_column_line(previous, block, profile):
        return True

    if _wide_math_lead_should_stand_alone(previous, block):
        return True

    if _looks_like_visual_label_after_prose(previous, block):
        return True

    if _STRONG_PARAGRAPH_START_RE.match(current_text) and not _previous_allows_indent_continuation(previous_text):
        return True

    if vertical_gap <= profile.soft_break_gap:
        return False

    if vertical_gap <= profile.hard_break_gap and _looks_like_continuation(previous_text, current_text):
        return False

    if vertical_gap > profile.hard_break_gap:
        return True

    if _STRONG_PARAGRAPH_START_RE.match(current_text):
        return True

    indent_delta = block.bbox.x0 - previous.bbox.x0
    if indent_delta > 42 and not _previous_allows_indent_continuation(previous_text):
        return True
    return False


def _looks_like_cross_page_continuation(
    previous: DocumentBlock,
    block: DocumentBlock,
    page_sizes: dict[int, tuple[float, float]],
) -> bool:
    if previous.page is None or block.page is None:
        return False
    try:
        previous_page = int(previous.page)
        current_page = int(block.page)
    except (TypeError, ValueError):
        return False
    if current_page != previous_page + 1:
        return False

    previous_text = previous.text.rstrip()
    current_text = block.text.strip()
    if not previous_text or not current_text:
        return False
    if _STRONG_PARAGRAPH_START_RE.match(current_text):
        return False
    if not _has_cross_page_continuation_signal(previous_text, current_text):
        return False

    if _near_page_boundary(previous, block, page_sizes):
        return True

    # Même sans dimensions de page, un tiret final à un changement de page
    # signale fortement une phrase coupée avant la page suivante.
    return previous_text.endswith("-")


def _has_cross_page_continuation_signal(previous_text: str, current_text: str) -> bool:
    if _looks_like_section_heading_text(previous_text) or _looks_like_section_heading_text(current_text):
        return False
    if previous_text.endswith("-"):
        return True
    if _previous_allows_indent_continuation(previous_text):
        return True
    if previous_text.endswith((".", "!", "?", "”", '"', "'")):
        return False
    return _looks_like_continuation(previous_text, current_text)


def _near_page_boundary(
    previous: DocumentBlock,
    block: DocumentBlock,
    page_sizes: dict[int, tuple[float, float]],
) -> bool:
    if previous.bbox is None or block.bbox is None or previous.page is None or block.page is None:
        return False

    previous_size = page_sizes.get(int(previous.page))
    current_size = page_sizes.get(int(block.page))
    if not previous_size or not current_size:
        return False

    previous_height = float(previous_size[1] or 0.0)
    current_height = float(current_size[1] or 0.0)
    if previous_height <= 0 or current_height <= 0:
        return False

    previous_near_end = previous.bbox.y1 >= previous_height * 0.72
    current_near_start = block.bbox.y0 <= current_height * 0.28
    return previous_near_end and current_near_start


def _looks_like_column_restart(
    previous: DocumentBlock,
    block: DocumentBlock,
    profile: _PageParagraphProfile,
) -> bool:
    if previous.bbox is None or block.bbox is None:
        return False
    vertical_regression = previous.bbox.y0 - block.bbox.y0
    if vertical_regression <= max(8.0, profile.line_height * 1.2):
        return False
    horizontal_shift = abs(block.bbox.center_x - previous.bbox.center_x)
    column_shift = horizontal_shift > max(80.0, min(previous.bbox.width, block.bbox.width) * 0.35)
    return column_shift


def _looks_like_reverse_column_jump(
    previous: DocumentBlock,
    block: DocumentBlock,
    profile: _PageParagraphProfile,
) -> bool:
    if previous.bbox is None or block.bbox is None:
        return False
    vertical_regression = previous.bbox.y0 - block.bbox.y0
    if vertical_regression <= max(8.0, profile.line_height * 1.2):
        return False
    left_shift = previous.bbox.center_x - block.bbox.center_x
    return left_shift > max(80.0, min(previous.bbox.width, block.bbox.width) * 0.35)


def _looks_like_column_flow_continuation(previous_text: str, current_text: str) -> bool:
    if not previous_text or not current_text:
        return False
    if previous_text.endswith((".", "!", "?", "”", '"', "'")):
        return False
    return previous_text.endswith("-") or _looks_like_continuation(previous_text, current_text)


def _looks_like_parallel_column_line(
    previous: DocumentBlock,
    block: DocumentBlock,
    profile: _PageParagraphProfile,
) -> bool:
    if previous.bbox is None or block.bbox is None:
        return False
    vertical_overlap = min(previous.bbox.y1, block.bbox.y1) - max(previous.bbox.y0, block.bbox.y0)
    same_text_band = (
        vertical_overlap > min(previous.bbox.height, block.bbox.height) * 0.35
        or abs(block.bbox.y0 - previous.bbox.y0) <= max(4.0, profile.line_height * 0.75)
    )
    if not same_text_band:
        return False
    horizontal_gap = max(block.bbox.x0 - previous.bbox.x1, previous.bbox.x0 - block.bbox.x1)
    if horizontal_gap <= max(18.0, profile.line_height * 1.5):
        return False
    horizontal_shift = abs(block.bbox.center_x - previous.bbox.center_x)
    return horizontal_shift > max(120.0, min(previous.bbox.width, block.bbox.width) * 0.65)


def _looks_like_adjacent_cross_column_line(
    previous: DocumentBlock,
    block: DocumentBlock,
    profile: _PageParagraphProfile,
) -> bool:
    if previous.bbox is None or block.bbox is None:
        return False
    if not _looks_like_cross_column_pair(previous.bbox, block.bbox, profile):
        return False
    vertical_regression = previous.bbox.y0 - block.bbox.y0
    if vertical_regression > max(50.0, profile.line_height * 4.0):
        return False
    vertical_gap = block.bbox.y0 - previous.bbox.y1
    same_or_next_band = (
        abs(block.bbox.y0 - previous.bbox.y0) <= max(18.0, profile.line_height * 1.25)
        or -profile.line_height <= vertical_gap <= profile.soft_break_gap
    )
    return same_or_next_band


def _looks_like_cross_column_pair(
    left: BoundingBox,
    right: BoundingBox,
    profile: _PageParagraphProfile,
) -> bool:
    horizontal_gap = max(right.x0 - left.x1, left.x0 - right.x1)
    if horizontal_gap <= max(18.0, profile.line_height * 1.5):
        return False
    horizontal_shift = abs(right.center_x - left.center_x)
    return horizontal_shift > max(120.0, min(left.width, right.width) * 0.65)


def _wide_math_lead_should_stand_alone(previous: DocumentBlock, block: DocumentBlock) -> bool:
    if previous.bbox is None or block.bbox is None:
        return False

    metadata = previous.metadata or {}
    if not (metadata.get("contains_inline_math") or metadata.get("formula_mode") in {"inline", "ambiguous"}):
        return False

    raw_type = str(metadata.get("raw_block_type") or "")
    if raw_type not in {"line_with_inline_math", "ambiguous_math_line"} and not _uses_math_font(metadata):
        return False

    previous_text = previous.text.strip()
    current_text = block.text.strip()
    if not previous_text or not current_text or len(previous_text) > 140:
        return False
    if _prose_word_count(previous_text) > 1:
        return False
    if _prose_word_count(current_text) < 2:
        return False
    if _math_signal_count(previous_text) < 2:
        return False

    return previous.bbox.width >= max(140.0, block.bbox.width * 0.45)


def _looks_like_visual_label_after_prose(previous: DocumentBlock, block: DocumentBlock) -> bool:
    if previous.bbox is None or block.bbox is None:
        return False
    previous_text = previous.text.strip()
    current_text = block.text.strip()
    if len(previous_text) < 45 or len(current_text) > 80:
        return False
    if _prose_word_count(current_text) > 4:
        return False
    if block.bbox.width > 170.0 or block.bbox.height > 36.0:
        return False
    if previous.bbox.width < 260.0:
        return False
    horizontal_shift = abs(block.bbox.center_x - previous.bbox.center_x)
    if horizontal_shift < 110.0:
        return False
    return previous_text.endswith((".", "!", "?", ":", ";")) or block.bbox.width <= previous.bbox.width * 0.42


def _looks_like_display_math_fragment(block: DocumentBlock) -> bool:
    text = block.text.strip()
    if block.bbox is None or not text or len(text) > 45:
        return False
    if not re.search(r"[$\\_^{}∑∫≈≤≥≠±√∼~]|[A-Za-z]_\{?", text):
        return False
    if len(re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", text)) > 1:
        return False
    metadata = block.metadata or {}
    if metadata.get("formula_mode") in {"inline", "ambiguous"}:
        return block.bbox.width < 120.0 and block.bbox.x0 > 120.0
    return block.bbox.width < 120.0 and block.bbox.x0 > 120.0


def _uses_math_font(metadata: dict) -> bool:
    font = str(metadata.get("font_name") or metadata.get("font") or "").casefold()
    return any(marker in font for marker in ("math", "cmmi", "cmsy", "cmex", "stix"))


def _prose_word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", text or ""))


def _math_signal_count(text: str) -> int:
    return len(re.findall(r"(?:\\[A-Za-z]+|[_^{}=<>]|[∼~≈→←⇒⇔∞≤≥≠±√∑∫α-ωΑ-Ω])", text or ""))


def _page_profiles(blocks: list[DocumentBlock]) -> dict[int, _PageParagraphProfile]:
    by_page: dict[int, list[DocumentBlock]] = {}
    for block in blocks:
        if block.type != "paragraph" or block.bbox is None or block.page is None:
            continue
        by_page.setdefault(int(block.page), []).append(block)

    profiles: dict[int, _PageParagraphProfile] = {}
    for page, page_blocks in by_page.items():
        ordered = sorted(page_blocks, key=lambda item: (item.bbox.y0, item.bbox.x0))  # type: ignore[union-attr]
        heights = [block.bbox.height for block in ordered if block.bbox and block.bbox.height > 0]
        gaps = [
            right.bbox.y0 - left.bbox.y1
            for left, right in zip(ordered, ordered[1:])
            if left.bbox
            and right.bbox
            and left.page == right.page
            and -2.0 <= right.bbox.y0 - left.bbox.y1 <= 36.0
        ]
        profiles[page] = _PageParagraphProfile(
            line_height=float(median(heights)) if heights else 12.0,
            normal_gap=float(median(gaps)) if gaps else 3.0,
        )
    return profiles


def _looks_like_continuation(previous_text: str, current_text: str) -> bool:
    if not current_text:
        return True
    if previous_text.endswith((",", ";", ":", "(", "[", "{", "+", "-", "=", "/", "→", "⇒")):
        return True
    if current_text[:1].islower():
        return True
    if current_text.startswith((",", ".", ";", ":", ")", "]", "}", "+", "-", "=", "/", "→", "⇒")):
        return True
    return bool(_CONNECTOR_START_RE.match(current_text)) and not _STRONG_PARAGRAPH_START_RE.match(current_text)


_SECTION_HEADING_TEXT_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Za-zÀ-ÿ0-9]")


def _looks_like_section_heading_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) > 140:
        return False
    if not _SECTION_HEADING_TEXT_RE.match(cleaned):
        return False
    if cleaned.endswith((".", "!", "?", ":", ";")):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’+-]*", cleaned)
    return 1 <= len(words) <= 12


def _previous_allows_indent_continuation(text: str) -> bool:
    return text.endswith((",", ";", ":", "(", "[", "{", "+", "-", "=", "/", "→", "⇒"))


def _join_hyphenated_line(left: str, right: str) -> str:
    left_base = left[:-1].rstrip()
    right_clean = right.lstrip()
    if not left_base or not right_clean:
        return left_base + right_clean

    left_word = re.search(r"([A-Za-zÀ-ÿ]{2,})$", left_base)
    right_word = re.match(r"([A-Za-zÀ-ÿ]{2,})", right_clean)
    if not left_word or not right_word:
        return left_base + right_clean

    prefix = left_word.group(1).casefold()
    suffix = right_word.group(1).casefold()

    if prefix.endswith("ing") and suffix.endswith("s"):
        return f"{left_base} {right_clean}"
    return left_base + right_clean


def _clean_merged_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _repair_soft_hyphenation(text)
    text = _LEADING_DISPLAY_EQUATION_RESIDUE_RE.sub("", text).strip()
    text = _LEADING_EQUATION_NUMBER_RE.sub("", text).strip()
    text = _ORPHAN_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


_PRESERVE_HYPHEN_PREFIXES = {
    "anti",
    "bi",
    "co",
    "cross",
    "few",
    "high",
    "inter",
    "intra",
    "low",
    "meta",
    "multi",
    "non",
    "one",
    "post",
    "pre",
    "semi",
    "self",
    "sub",
    "super",
    "two",
    "zero",
}
_SOFT_HYPHEN_SUFFIXES = (
    "able",
    "al",
    "ance",
    "ary",
    "ate",
    "ation",
    "ed",
    "ence",
    "ent",
    "er",
    "es",
    "ful",
    "ible",
    "ic",
    "ing",
    "ion",
    "ity",
    "ive",
    "less",
    "ly",
    "ment",
    "ness",
    "ory",
    "ous",
    "sion",
    "tion",
    "ual",
)


def _repair_soft_hyphenation(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        left = match.group(1)
        right = match.group(2)
        lower_left = left.casefold()
        lower_right = right.casefold()
        if lower_left in _PRESERVE_HYPHEN_PREFIXES:
            return f"{left}-{right}"
        if len(left) <= 5 or lower_right.endswith(_SOFT_HYPHEN_SUFFIXES):
            return f"{left}{right}"
        return f"{left}-{right}"

    return re.sub(r"\b([A-Za-zÀ-ÿ]{2,})-\s+([a-zà-ÿ]{2,})\b", replace, text)


def _repair_split_heading_prefixes(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    repaired: list[DocumentBlock] = []
    index = 0
    while index < len(blocks):
        if index + 1 >= len(blocks):
            repaired.append(blocks[index])
            index += 1
            continue

        current = blocks[index]
        next_block = blocks[index + 1]
        if _looks_like_split_heading_prefix(current, next_block):
            match = _TRAILING_HEADING_PREFIX_RE.match((current.text or "").strip())
            assert match is not None
            current.text = match.group("body").strip()
            next_block.text = f"{match.group('prefix').rstrip('.')} {next_block.text.strip()}"
        repaired.append(current)
        index += 1
    return repaired


def _looks_like_split_heading_prefix(current: DocumentBlock, next_block: DocumentBlock) -> bool:
    if current.type != "paragraph" or next_block.type not in {"heading", "subheading", "subsubheading"}:
        return False
    if current.page != next_block.page or current.bbox is None or next_block.bbox is None:
        return False
    text = (current.text or "").strip()
    match = _TRAILING_HEADING_PREFIX_RE.match(text)
    if match is None:
        return False
    body = match.group("body")
    if len(body) < 40:
        return False
    if len(next_block.text.strip()) < 4:
        return False
    vertical_gap = next_block.bbox.y0 - current.bbox.y1
    return vertical_gap <= 18.0
