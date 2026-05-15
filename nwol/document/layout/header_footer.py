from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import median

from document.models import RawLine


def normalize_repeated_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_repeated_headers_footers(
    lines: list[RawLine],
    page_sizes: dict[int, tuple[float, float]],
    threshold: float = 0.45,
    margin_ratio: float = 0.12,
) -> set[str]:
    pages = set(page_sizes) or {line.page for line in lines}
    if not pages:
        return set()
    if len(pages) < 3:
        return set()

    candidates_by_page: dict[int, set[str]] = defaultdict(set)
    for line in lines:
        _, height = page_sizes.get(line.page, (0.0, 0.0))
        if height <= 0:
            continue
        in_top = line.bbox.y0 <= height * margin_ratio
        in_bottom = line.bbox.y1 >= height * (1.0 - margin_ratio)
        if not in_top and not in_bottom:
            continue
        key = normalize_repeated_text(line.text)
        if len(key) >= 3:
            candidates_by_page[line.page].add(key)

    counter: Counter[str] = Counter()
    for keys in candidates_by_page.values():
        counter.update(keys)

    total_pages = max(len(pages), 1)
    return {key for key, count in counter.items() if count / total_pages >= threshold}


def remove_repeated_headers_footers(
    lines: list[RawLine],
    page_sizes: dict[int, tuple[float, float]],
    repeated: set[str] | None = None,
    threshold: float = 0.45,
    margin_ratio: float = 0.12,
) -> tuple[list[RawLine], set[str]]:
    repeated = repeated or detect_repeated_headers_footers(
        lines,
        page_sizes,
        threshold=threshold,
        margin_ratio=margin_ratio,
    )
    if not repeated:
        return _remove_first_page_preamble(lines, page_sizes), set()

    filtered: list[RawLine] = []
    removed: set[str] = set()
    for line in lines:
        _, height = page_sizes.get(line.page, (0.0, 0.0))
        key = normalize_repeated_text(line.text)
        in_margin = height > 0 and (
            line.bbox.y0 <= height * margin_ratio
            or line.bbox.y1 >= height * (1.0 - margin_ratio)
        )
        if in_margin and key in repeated:
            removed.add(key)
            continue
        filtered.append(line)
    return _remove_first_page_preamble(filtered, page_sizes), removed


def _remove_first_page_preamble(
    lines: list[RawLine],
    page_sizes: dict[int, tuple[float, float]],
) -> list[RawLine]:
    if not lines or 1 not in page_sizes:
        return lines[:]

    width, height = page_sizes.get(1, (0.0, 0.0))
    if height <= 0:
        return lines[:]

    page_lines = [line for line in lines if line.page == 1]
    if len(page_lines) < 4:
        return lines[:]

    body_size = median([line.font_size for line in page_lines if line.font_size] or [11.0])
    top_limit = height * 0.08
    search_limit = height * 0.18
    main_heading = None
    for line in sorted(page_lines, key=lambda item: item.bbox.y0):
        size = line.font_size or body_size
        text = line.text.strip()
        if line.bbox.y0 < top_limit:
            continue
        if line.bbox.y0 > search_limit:
            break
        is_large = size >= max(15.0, body_size * 1.15)
        is_heading_like = bool(re.match(r"^\d+(?:\.\s*)?\s+", text)) or line.bbox.width > width * 0.45
        if is_large and is_heading_like:
            main_heading = line
            break

    if main_heading is None:
        return lines[:]

    cutoff = main_heading.bbox.y0 - 4.0
    return [
        line
        for line in lines
        if not (line.page == 1 and line.bbox.y1 < cutoff)
    ]
