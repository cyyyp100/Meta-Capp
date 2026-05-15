from __future__ import annotations

import re
from typing import Any


HEADING_BLOCK_TYPES = {"heading", "subheading", "subsubheading"}

_SECTION_NUMBER_RE = re.compile(
    r"^\s*(?P<number>(?:\d+\s*\.\s*)*\d+|[A-Z](?:\s*\.\s*\d+)+|[A-Z]\.)"
    r"(?:\.)?(?:\s+|(?=[A-Za-zÀ-ÿ]))\S+",
    re.I,
)
_SECTION_NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?P<number>(?:\d+\s*\.\s*)*\d+|[A-Z](?:\s*\.\s*\d+)+|[A-Z]\.)"
    r"(?:\.)?(?:\s+|(?=[A-Za-zÀ-ÿ]))(?P<label>.*)$",
    re.I,
)


def normalize_heading_title(title: str) -> str:
    clean = _clean_heading_title(str(title or ""))
    number = section_number(clean)
    if number:
        match = _SECTION_NUMBER_PREFIX_RE.match(clean)
        label = match.group("label") if match else clean
        label = re.sub(r"^[\s.\-–—:]+", "", label).strip()
        return re.sub(r"\s+", " ", f"{number} {label}").strip().casefold()
    return re.sub(r"\s+", " ", clean).strip().casefold()


def heading_titles_match(candidate: str, wanted_title: str) -> bool:
    """Match a visible heading against a TOC title, tolerating missing numbers.

    Some PDF TOCs expose "A Simple Climate Equation" while page extraction sees
    "5.1 A Simple Climate Equation". Exact normalization would miss that match
    and start the reader at the beginning of the page.
    """
    candidate_norm = normalize_heading_title(candidate)
    wanted_norm = normalize_heading_title(wanted_title)
    if not candidate_norm or not wanted_norm:
        return False
    if candidate_norm == wanted_norm:
        return True

    candidate_label = _normalize_heading_label_only(candidate)
    wanted_label = _normalize_heading_label_only(wanted_title)
    if not candidate_label or not wanted_label:
        return False
    if candidate_label == wanted_label:
        return True
    return _same_numbered_heading_with_embedded_body(candidate, wanted_title, candidate_label, wanted_label)


def section_number(title: str) -> str | None:
    match = _SECTION_NUMBER_RE.match(_clean_heading_title(str(title or "")))
    if not match:
        return None
    return _normalize_section_number(match.group("number"))


def section_number_key(title: str) -> tuple[Any, ...] | None:
    number = section_number(title)
    if not number:
        return None
    key: list[Any] = []
    for part in number.split("."):
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.casefold())
    return tuple(key)


def _clean_heading_title(title: str) -> str:
    clean = re.sub(r"\s+", " ", str(title or "")).strip()
    clean = re.sub(r"^\s*[\.\-–—•·]+\s*(?=[A-Za-zÀ-ÿ])", "", clean).strip()
    return clean


def _normalize_heading_label_only(title: str) -> str:
    clean = _clean_heading_title(str(title or ""))
    match = _SECTION_NUMBER_PREFIX_RE.match(clean)
    if match:
        clean = match.group("label")
    clean = re.sub(r"^[\s.\-–—:]+", "", clean).strip()
    return re.sub(r"\s+", " ", clean).strip().casefold()


def _same_numbered_heading_with_embedded_body(
    candidate: str,
    wanted_title: str,
    candidate_label: str,
    wanted_label: str,
) -> bool:
    candidate_number = section_number(candidate)
    wanted_number = section_number(wanted_title)
    if not candidate_number or candidate_number != wanted_number:
        return False
    if not wanted_label.startswith(candidate_label + " "):
        return False
    remainder = wanted_label[len(candidate_label):].strip()
    words = remainder.split()
    if len(words) < 5:
        return False
    first_words = " ".join(words[:10])
    return bool(
        re.search(
            r"\b(can|may|must|should|will|is|are|was|were|means|depends|represents?|describes?|"
            r"shows?|uses?|requires?|allows?|gives?|has|have|does|do|peut|peuvent|est|sont|"
            r"signifie|depend|dépend|represente|représente)\b",
            first_words,
            re.I,
        )
    )


def _normalize_section_number(number: str) -> str:
    raw = str(number or "").strip().rstrip(".")
    if not raw:
        return ""
    if re.fullmatch(r"[A-Z]\.?", raw, re.I):
        return raw.rstrip(".")
    parts = [part for part in re.split(r"\s*\.\s*", raw) if part]
    return ".".join(parts)


def child_sections_for_chapter(chapter: dict, chapters: list[dict]) -> list[dict]:
    """Return selectable child headings for a chapter.

    Scientific PDFs often put two numbered sections on the same page. Page-only
    grouping can therefore attach "3.2" to chapter "4". Numbered headings are
    grouped by their section prefix first, with the old page-range behavior kept
    as fallback for unnumbered documents.
    """
    chapter_level = _chapter_level(chapter)
    parent_number = section_number(str(chapter.get("title") or ""))
    if parent_number and "." not in parent_number:
        parent_prefix = parent_number + "."
        children = [
            c
            for c in chapters
            if _chapter_level(c) > chapter_level
            and (section_number(str(c.get("title") or "")) or "").startswith(parent_prefix)
        ]
        return _dedupe_numbered_sections(children)

    ch_start = _chapter_page(chapter, "page_start", 1)
    ch_end = _next_peer_page(chapter, chapters) or float("inf")
    return [
        c
        for c in chapters
        if _chapter_level(c) > chapter_level
        and ch_start <= _chapter_page(c, "page_start", 1) < ch_end
    ]


def extraction_end_page_for_scope(chapter: dict, chapters: list[dict], total_pages: int) -> int:
    page_start = _chapter_page(chapter, "page_start", 1)
    page_end = _chapter_page(chapter, "page_end", total_pages)
    if page_end < page_start:
        page_end = total_pages

    target_key = section_number_key(str(chapter.get("title") or ""))
    target_level = _chapter_level(chapter)
    if target_key is None:
        return min(total_pages, max(page_start, page_end))

    next_page: int | None = None
    for candidate in chapters:
        if candidate is chapter or _same_chapter(candidate, chapter):
            continue
        cand_key = section_number_key(str(candidate.get("title") or ""))
        if cand_key is None or not _section_key_after(cand_key, target_key):
            continue
        if _chapter_level(candidate) > target_level:
            continue
        cand_page = _chapter_page(candidate, "page_start", page_start)
        if next_page is None or cand_page < next_page:
            next_page = cand_page

    if next_page is not None:
        page_end = max(page_end, next_page)
    return min(total_pages, max(page_start, page_end))


def slice_blocks_for_heading_scope(blocks: list[dict], chapter: dict) -> list[dict]:
    if not blocks:
        return blocks

    if chapter.get("page_end") is not None:
        blocks = _drop_non_heading_blocks_after_page_end(blocks, _chapter_page(chapter, "page_end", 1))

    title = str(chapter.get("title") or "")
    start = _find_heading_index(blocks, title)
    if start is None:
        return _fallback_slice_for_missing_heading(blocks, chapter)

    target_number = section_number(title)
    target_level = _chapter_level(chapter)
    if not target_number:
        return _contiguous_heading_slice(blocks, start, target_level)

    return _numbered_heading_slice(blocks, start, target_number, target_level)


def slice_blocks_from_heading_to_end(blocks: list[dict], chapter: dict) -> list[dict]:
    """Find the chapter heading start position and return all blocks from there to the end.

    Unlike slice_blocks_for_heading_scope, this does NOT stop at chapter boundaries
    and does NOT truncate at chapter["page_end"]. Used for full-PDF reading sessions
    where the user reads from a chosen chapter to the very last page.
    """
    if not blocks:
        return blocks

    title = str(chapter.get("title") or "")
    start = _find_heading_index(blocks, title)
    if start is not None:
        return _slice_from_index_to_end(blocks, start)

    # Fallback: try to locate heading by section number
    target_number = section_number(title)
    if target_number:
        idx = _find_first_in_scope_heading_index(blocks, target_number)
        if idx is not None:
            return _slice_from_index_to_end(blocks, idx)

    # Last resort: drop blocks before the chapter's start page
    page_start = _chapter_page(chapter, "page_start", 1)
    return _drop_blocks_before_page(blocks, page_start) or blocks


def _slice_from_index_to_end(blocks: list[dict], start: int) -> list[dict]:
    """Return blocks from a heading onward, preserving same-page right columns.

    Some PDF extractors surface a page in geometric y-order when a full-width
    figure/caption sits above two columns. In that case a heading at the bottom
    of the left column can appear after right-column blocks in the raw list. A
    "read from here" scope must still consume the rest of the current page's
    right column before advancing to the next page.
    """
    selected = list(blocks[start:])
    if not selected:
        return selected

    start_block = blocks[start]
    start_page = _block_page(start_block)
    start_bbox = _block_bbox(start_block)
    if start_page is None or start_bbox is None:
        return selected

    page_width = _block_page_width(start_block) or _infer_page_width(blocks, start_page)
    if page_width is None:
        return selected

    right_floor = max(page_width * 0.48, start_bbox[2] + 12.0)
    if start_bbox[0] >= right_floor or start_bbox[2] >= page_width * 0.82:
        return selected

    same_page_tail: list[dict] = []
    later_tail: list[dict] = []
    for block in selected:
        if _block_page(block) == start_page:
            same_page_tail.append(block)
        else:
            later_tail.append(block)

    if not same_page_tail:
        return selected

    existing_ids = {_stable_block_identity(block) for block in same_page_tail}
    backfill = [
        block
        for block in blocks[:start]
        if _stable_block_identity(block) not in existing_ids
        and _block_page(block) == start_page
        and _is_right_column_block(block, right_floor, page_width)
    ]
    if not backfill:
        return selected

    left_tail = [block for block in same_page_tail if not _is_right_column_block(block, right_floor, page_width)]
    right_tail = [block for block in same_page_tail if _is_right_column_block(block, right_floor, page_width)]
    right_tail.extend(backfill)
    right_tail = _dedupe_blocks(sorted(right_tail, key=_same_page_position_key))
    return [*left_tail, *right_tail, *later_tail]


def heading_search_start_page(chapter: dict) -> int:
    """Return the first page to extract when locating a chapter heading.

    PDF TOCs are sometimes off by one page when the selected heading sits at the
    bottom of the previous page. For numbered sections we extract one page
    earlier, then slice by heading so previous material is discarded.
    """
    page_start = _chapter_page(chapter, "page_start", 1)
    title = str(chapter.get("title") or "")
    if section_number(title):
        return max(1, page_start - 1)
    return page_start


def _numbered_heading_slice(
    blocks: list[dict],
    start: int,
    target_number: str,
    target_level: int,
) -> list[dict]:
    selected: list[dict] = []
    in_scope = False
    started = False
    start_block = blocks[start]
    start_page = _block_page(start_block)
    column_floor_x: float | None = None

    for block in blocks[start:]:
        if _is_heading_block(block):
            text = str(block.get("text") or "")
            current_number = section_number(text)
            current_level = _block_heading_level(block)

            if current_number:
                if _number_belongs_to_scope(current_number, target_number):
                    in_scope = True
                    started = True
                    selected.append(block)
                    continue

                if started and current_level <= target_level:
                    if _looks_like_previous_column_heading(block, start_block, current_number, target_number):
                        column_floor_x = _right_column_floor(start_block)
                        continue
                    if _number_after(current_number, target_number):
                        break

                # Same page, two-column PDFs may surface a previous numbered
                # subsection after the selected heading. Skip that foreign
                # subsection until a heading belonging to this scope appears.
                if started:
                    in_scope = False
                    continue

            elif started and current_level <= target_level:
                if (
                    in_scope
                    and column_floor_x is not None
                    and not _is_left_of_column_floor(block, start_page, column_floor_x)
                ):
                    selected.append({**block, "type": "paragraph"})
                    continue
                break
            elif started and in_scope:
                selected.append({**block, "type": "paragraph"})
                continue

        if in_scope:
            if column_floor_x is not None and _is_left_of_column_floor(block, start_page, column_floor_x):
                continue
            selected.append(block)

    return selected or blocks[start:]


def _fallback_slice_for_missing_heading(blocks: list[dict], chapter: dict) -> list[dict]:
    title = str(chapter.get("title") or "")
    target_number = section_number(title)
    target_level = _chapter_level(chapter)

    if target_number:
        start = _find_first_in_scope_heading_index(blocks, target_number)
        if start is not None:
            return _numbered_heading_slice(blocks, start, target_number, target_level)

    page_start = _chapter_page(chapter, "page_start", 1)
    page_filtered = _drop_blocks_before_page(blocks, page_start)
    trimmed = _trim_leading_continuation_blocks(page_filtered)
    return trimmed or page_filtered or blocks


def _find_first_in_scope_heading_index(blocks: list[dict], target_number: str) -> int | None:
    for idx, block in enumerate(blocks):
        if not _is_heading_block(block):
            continue
        current_number = section_number(str(block.get("text") or ""))
        if current_number and _number_belongs_to_scope(current_number, target_number):
            return idx
    return None


def _drop_blocks_before_page(blocks: list[dict], page_start: int) -> list[dict]:
    filtered = [
        block
        for block in blocks
        if _block_page(block) is None or int(_block_page(block) or page_start) >= page_start
    ]
    return filtered or blocks


def _drop_non_heading_blocks_after_page_end(blocks: list[dict], page_end: int) -> list[dict]:
    return [
        block
        for block in blocks
        if _block_page(block) is None or int(_block_page(block) or page_end) <= page_end or _is_heading_block(block)
    ]


def _trim_leading_continuation_blocks(blocks: list[dict]) -> list[dict]:
    start = 0
    while start < len(blocks):
        block = blocks[start]
        if _is_heading_block(block):
            break
        text = str(block.get("text") or block.get("latex") or "").strip()
        if not text:
            start += 1
            continue
        if not _looks_like_continuation_start(text):
            break
        start += 1
    return blocks[start:]


def _looks_like_continuation_start(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", str(text or "")).strip()
    if not stripped:
        return True
    first = stripped[:1]
    first_word = re.match(r"[A-Za-zÀ-ÿ]+", stripped)
    if first_word and first_word.group(0)[:1].islower():
        return True
    if first in ",.;:)]}" or stripped.startswith("-"):
        return True
    if re.match(r"^(?:ing|ed|tion|sion|ment|jects?|objects?|tups?)\b", stripped, re.I):
        return True
    return False


def _contiguous_heading_slice(blocks: list[dict], start: int, target_level: int) -> list[dict]:
    end = len(blocks)
    start_block = blocks[start]
    for idx in range(start + 1, len(blocks)):
        block = blocks[idx]
        if _is_heading_block(block) and _block_heading_level(block) <= target_level:
            if _is_adjacent_duplicate_title_fragment(start_block, block):
                continue
            end = idx
            break
    return blocks[start:end]


def _find_heading_index(blocks: list[dict], title: str) -> int | None:
    wanted_number = section_number(title)
    for idx, block in enumerate(blocks):
        if not _is_heading_block(block):
            continue
        text = str(block.get("text") or "")
        if heading_titles_match(text, title):
            return idx
        if wanted_number and section_number(text) == wanted_number:
            return idx
    return None


def _number_belongs_to_scope(number: str, target: str) -> bool:
    return number == target or number.startswith(target + ".")


def _number_after(number: str, target: str) -> bool:
    current_key = _number_key(number)
    target_key = _number_key(target)
    if current_key is None or target_key is None:
        return False
    return _section_key_after(current_key, target_key)


def _number_before(number: str, target: str) -> bool:
    current_key = _number_key(number)
    target_key = _number_key(target)
    if current_key is None or target_key is None:
        return False
    return _section_key_after(target_key, current_key)


def _number_key(number: str) -> tuple[Any, ...] | None:
    normalized = _normalize_section_number(number)
    if not normalized:
        return None
    key: list[Any] = []
    for part in normalized.split("."):
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.casefold())
    return tuple(key)


def _looks_like_previous_column_heading(
    block: dict,
    start_block: dict,
    current_number: str,
    target_number: str,
) -> bool:
    if not _number_before(current_number, target_number):
        return False
    if _block_page(block) != _block_page(start_block):
        return False
    block_bbox = _block_bbox(block)
    start_bbox = _block_bbox(start_block)
    if block_bbox is None or start_bbox is None:
        return False
    return block_bbox[2] < start_bbox[0] - 12.0


def _right_column_floor(start_block: dict) -> float | None:
    start_bbox = _block_bbox(start_block)
    if start_bbox is None:
        return None
    return start_bbox[0] - 12.0


def _is_left_of_column_floor(block: dict, start_page: int | None, floor_x: float) -> bool:
    if start_page is None or _block_page(block) != start_page:
        return False
    bbox = _block_bbox(block)
    if bbox is None:
        return False
    center_x = (bbox[0] + bbox[2]) / 2.0
    return center_x < floor_x


def _block_bbox(block: dict) -> tuple[float, float, float, float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox[:4])
    except (TypeError, ValueError):
        return None
    return x0, y0, x1, y1


def _block_page_width(block: dict) -> float | None:
    metadata = block.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    for value in (
        metadata.get("page_width"),
        block.get("page_width"),
    ):
        try:
            width = float(value)
        except (TypeError, ValueError):
            continue
        if width > 0:
            return width
    return None


def _infer_page_width(blocks: list[dict], page: int) -> float | None:
    widths: list[float] = []
    for block in blocks:
        if _block_page(block) != page:
            continue
        width = _block_page_width(block)
        if width:
            widths.append(width)
            continue
        bbox = _block_bbox(block)
        if bbox is not None:
            widths.append(float(bbox[2]))
    return max(widths) if widths else None


def _is_right_column_block(block: dict, right_floor: float, page_width: float) -> bool:
    bbox = _block_bbox(block)
    if bbox is None:
        return False
    if bbox[0] < right_floor:
        return False
    if bbox[2] <= bbox[0]:
        return False
    metadata = block.get("metadata") or {}
    if isinstance(metadata, dict) and (
        metadata.get("is_metadata")
        or metadata.get("is_reference")
        or metadata.get("is_header_footer")
    ):
        return False
    if block.get("is_metadata") or block.get("is_reference"):
        return False
    return (bbox[2] - bbox[0]) <= page_width * 0.58 or bbox[0] >= page_width * 0.55


def _same_page_position_key(block: dict) -> tuple[float, float]:
    bbox = _block_bbox(block)
    if bbox is None:
        return (float("inf"), float("inf"))
    return (bbox[1], bbox[0])


def _stable_block_identity(block: dict) -> tuple[str, object] | tuple[str, int]:
    block_id = block.get("id")
    if block_id is not None:
        return ("id", block_id)
    return ("object", id(block))


def _dedupe_blocks(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[tuple[str, object] | tuple[str, int]] = set()
    for block in blocks:
        key = _stable_block_identity(block)
        if key in seen:
            continue
        seen.add(key)
        result.append(block)
    return result


def _is_adjacent_duplicate_title_fragment(start_block: dict, block: dict) -> bool:
    """Treat semantic full-title headings and geometric title fragments as one heading.

    Fusion can put an invisible semantic title before the visible lines extracted
    geometrically. Without this guard, unnumbered scopes stop immediately at the
    first visible title line and contain only the invisible block.
    """
    if _block_page(block) != _block_page(start_block):
        return False
    start_raw = str(start_block.get("text") or "")
    raw = str(block.get("text") or "")
    start_text = normalize_heading_title(start_raw)
    text = normalize_heading_title(raw)
    if not start_text or not text:
        return False
    return heading_titles_match(raw, start_raw) or text in start_text or start_text in text


def _section_key_after(candidate: tuple[Any, ...], target: tuple[Any, ...]) -> bool:
    try:
        return candidate > target
    except TypeError:
        return False


def _is_heading_block(block: dict) -> bool:
    return str(block.get("type") or "") in HEADING_BLOCK_TYPES


def _block_heading_level(block: dict) -> int:
    try:
        return min(3, max(1, int(block.get("level") or 1)))
    except (TypeError, ValueError):
        btype = str(block.get("type") or "heading")
        return {"heading": 1, "subheading": 2, "subsubheading": 3}.get(btype, 1)


def _block_page(block: dict) -> int | None:
    for key in ("page_number", "page_start", "page"):
        try:
            value = block.get(key)
            if value is not None:
                return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return None


def _chapter_level(chapter: dict) -> int:
    try:
        return min(3, max(1, int(chapter.get("toc_level") or 1)))
    except (TypeError, ValueError):
        return 1


def _chapter_page(chapter: dict, key: str, default: int) -> int:
    try:
        return max(1, int(chapter.get(key) or default))
    except (TypeError, ValueError):
        return default


def _next_peer_page(chapter: dict, chapters: list[dict]) -> int | None:
    start = _chapter_page(chapter, "page_start", 1)
    level = _chapter_level(chapter)
    pages = [
        _chapter_page(c, "page_start", start)
        for c in chapters
        if not _same_chapter(c, chapter)
        and _chapter_level(c) <= level
        and _chapter_page(c, "page_start", start) >= start
    ]
    return min(pages) if pages else None


def _same_chapter(left: dict, right: dict) -> bool:
    left_id = left.get("id")
    right_id = right.get("id")
    if left_id is not None and right_id is not None:
        return left_id == right_id
    return (
        normalize_heading_title(str(left.get("title") or ""))
        == normalize_heading_title(str(right.get("title") or ""))
        and _chapter_page(left, "page_start", 1) == _chapter_page(right, "page_start", 1)
        and _chapter_level(left) == _chapter_level(right)
    )


def _dedupe_numbered_sections(chapters: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen_numbers: set[str] = set()
    for chapter in chapters:
        number = section_number(str(chapter.get("title") or ""))
        if number:
            if number in seen_numbers:
                continue
            seen_numbers.add(number)
        result.append(chapter)
    return result
