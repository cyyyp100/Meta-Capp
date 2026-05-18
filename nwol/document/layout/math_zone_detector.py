from __future__ import annotations

import re

from document.layout.column_detector import detect_columns
from document.layout.reading_order import order_page_blocks
from document.models import BoundingBox, RawBlock, RawLine


MATH_CHARS = set(
    "∼→←⇒⇔∞≤≥≠±√∑∏∫αβγδελμνπσφθℓ⋅·"
    "=<>^_{}"
)
STRONG_MATH_CHARS = set(
    "∼→←⇒⇔∞≤≥≠±√∑∏∫αβγδελμνπσφθℓ⋅·"
    "=<>^_{}"
)

PROSE_RE = re.compile(
    r"\b(soit|suite|fonction|définition|definition|proposition|exemple|"
    r"théorème|theoreme|démonstration|demonstration|remarque|piège|piege|"
    r"comme|donc|car|avec|pour|alors|on|obtient|partir|voisinage|"
    r"nombres?|réels?|reels?|entiers?|naturels?|tout|tous|toute|toutes|"
    r"écrire|ecrire|revient|autrement|dit|équivalent|equivalent|"
    r"choisi|simplifier|expression|ordre|principal|relation|réciproque|reciproque|"
    r"constantes|multiplicatives|comptent|éviter|eviter|garder|terme|limite)\b",
    re.I,
)


def union_bbox(lines: list[RawLine]) -> BoundingBox:
    bbox = lines[0].bbox
    for line in lines[1:]:
        bbox = bbox.union(line.bbox)
    return bbox


def line_math_score(line: RawLine) -> float:
    text = line.text.strip()
    if not text:
        return 0.0

    math_hits = sum(1 for c in text if c in MATH_CHARS)
    symbols = sum(not c.isalnum() and not c.isspace() for c in text)
    digits = sum(c.isdigit() for c in text)
    prose_words = len(re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", text))

    score = 0.0
    score += min(3.0, math_hits * 0.8)
    score += min(2.0, symbols * 0.4)
    score += min(1.0, digits * 0.15)

    if len(text) <= 20:
        score += 0.8
    if len(text) <= 8:
        score += 0.8

    if _line_uses_math_font(line) and re.search(r"[=<>≤≥≠≈∼~]", text):
        score += 1.2

    if PROSE_RE.search(text):
        score -= 2.0
    if prose_words >= 3:
        score -= 1.5

    return score


def looks_like_math_fragment(line: RawLine) -> bool:
    text = line.text.strip()
    if _looks_like_non_math_label(text):
        return False
    if not _has_math_signal(line, text):
        return False
    if _looks_like_section_number(text):
        return False
    return line_math_score(line) >= 1.2


def _has_math_signal(line: RawLine, text: str) -> bool:
    if any(char in STRONG_MATH_CHARS for char in text):
        return True
    if re.search(r"(?<=[A-Za-z0-9)\]}])\s*(?:[+\-*/])\s*(?=[A-Za-z0-9({\[])", text):
        return True
    if re.search(r"(?<![A-Za-z])\d+\s*/\s*\d+(?![A-Za-z])", text):
        return True
    if re.search(r"(?<=[A-Za-z0-9)\]}])\s*[⋅·]\s*(?=[A-Za-z0-9({\[])", text):
        return True
    if re.fullmatch(r"n!", text.strip(), re.I):
        return True
    if re.fullmatch(r"[uvw](?:n|_\{?n\}?)?", text.strip(), re.I):
        return True
    if re.fullmatch(r"[\])}]+\s*[A-Za-z](?:[_^]\{?[A-Za-z0-9]+\}?)?", text.strip()):
        return True
    if _line_uses_math_font(line) and re.fullmatch(r"[A-Za-z0-9.,;:(){}\[\]\s+\-*/=<>]+", text):
        return True
    return False


def _line_uses_math_font(line: RawLine) -> bool:
    fonts = [line.font_name or ""]
    fonts.extend(span.font_name or "" for span in getattr(line, "spans", []) if getattr(span, "font_name", None))
    return any(
        any(marker in font.casefold() for marker in ("math", "cmmi", "cmsy", "cmex", "stix", "lmmath"))
        for font in fonts
    )


def _looks_like_section_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)+\.?", text.strip()))


def _looks_like_non_math_label(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text.strip())
    if not stripped:
        return True
    if re.search(r"@|https?://|www\.|doi\.org", stripped, re.I):
        return True
    if re.fullmatch(r"\[?(?:CrossRef|PubMed|Google Scholar)\]?", stripped, re.I):
        return True
    if re.fullmatch(r"[A-Z][A-Za-z]*(?:-[A-Za-z0-9]+)+(?:\s+[A-Za-z][A-Za-z0-9]+)*", stripped):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+){1,3}", stripped):
        return True
    if re.match(r"^\d+(?:\.\d+)+\.?\s+[A-Za-zÀ-ÿ]", stripped):
        return True
    if re.match(r"^\d+\.\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ0-9'’(),:;\-\s]{2,90}$", stripped):
        return True
    if re.search(r"\b(?:P\.O\.|Box|Lebanon|Saudi Arabia|University|College|Department)\b", stripped, re.I):
        return True
    return False


def vertical_gap(a: RawLine, b: RawLine) -> float:
    return b.bbox.y0 - a.bbox.y1


def horizontal_overlap(a: RawLine, b: RawLine) -> float:
    return min(a.bbox.x1, b.bbox.x1) - max(a.bbox.x0, b.bbox.x0)


def can_group_math_lines(a: RawLine, b: RawLine) -> bool:
    if a.page != b.page:
        return False

    gap = vertical_gap(a, b)
    overlap = horizontal_overlap(a, b)
    vertical_overlap = min(a.bbox.y1, b.bbox.y1) - max(a.bbox.y0, b.bbox.y0)

    near_vertically = (
        vertical_overlap >= min(a.bbox.height, b.bbox.height) * 0.35
        or -4 <= gap <= max(22.0, a.bbox.height * 2.5)
    )
    aligned = overlap > -35

    return near_vertically and aligned


def classify_math_group(lines: list[RawLine], page_width: float | None = None) -> str:
    """Return one of: inline, display, ambiguous."""
    if not lines:
        return "ambiguous"

    text = " ".join(line.text.strip() for line in lines if line.text.strip())
    bbox = union_bbox(lines)
    avg_len = sum(len(line.text.strip()) for line in lines) / max(len(lines), 1)
    math_score = sum(line_math_score(line) for line in lines)
    prose_words = len(re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", text))
    prose = bool(PROSE_RE.search(text)) or prose_words >= 3

    if _looks_like_unclosed_inline_tail(lines):
        return "ambiguous"

    if len(lines) >= 2 and any(_looks_like_equation_number(line.text) for line in lines):
        if any(_looks_like_centered_display_math_line(line, page_width, line_math_score(line)) for line in lines):
            return "display"

    # A single normal line with prose around math is inline.
    if len(lines) == 1:
        if prose or len(text) > 45:
            if _looks_like_centered_display_math_line(lines[0], page_width, math_score):
                return "display"
            return "inline"
        if _looks_like_incomplete_inline_piece(text):
            return "ambiguous"
        if _looks_like_centered_display_math_line(lines[0], page_width, math_score):
            return "display"
        if page_width and page_width > 0:
            center_distance = abs(bbox.center_x - page_width / 2.0)
            if center_distance < page_width * 0.22 and math_score >= 2.5 and len(text) <= 50:
                return "display"
        if math_score >= 2.5 and len(text) <= 40:
            return "display"
        return "inline"

    # Multiple short stacked lines are likely a display formula.
    if len(lines) >= 2 and avg_len <= 18 and not prose:
        return "display"

    # Centered formula block.
    if page_width and page_width > 0:
        center_distance = abs(bbox.center_x - page_width / 2.0)
        if center_distance < page_width * 0.20 and math_score >= 3.0 and not prose:
            return "display"

    # Paragraph-like content with math should remain inline/ambiguous.
    if prose or len(text) > 80:
        return "inline"

    return "ambiguous"


def _looks_like_centered_display_math_line(line: RawLine, page_width: float | None, math_score: float) -> bool:
    if not page_width or page_width <= 0:
        return False
    text = line.text.strip()
    if len(text) > 90:
        return False
    if not _line_uses_math_font(line):
        return False
    if not re.search(r"[=<>≤≥≠≈∼~]|[+−\-*/×]", text):
        return False
    center_distance = abs(line.bbox.center_x - page_width / 2.0)
    return center_distance < page_width * 0.24 and math_score >= 1.4


def _looks_like_unclosed_inline_tail(lines: list[RawLine]) -> bool:
    texts = [line.text.strip() for line in lines if line.text.strip()]
    if len(texts) < 2:
        return False
    joined = " ".join(texts)
    if not re.search(r"^[A-Za-z0-9\s+\-*/().,]+$", joined):
        return False
    has_relation = any(symbol in joined for symbol in ("=", "→", "∼", "~", "<", ">"))
    has_unmatched_closing = (
        joined.count(")") > joined.count("(")
        or joined.count("]") > joined.count("[")
        or joined.count("}") > joined.count("{")
    )
    return not has_relation and (
        has_unmatched_closing
        or any(text.startswith((")", "]", "}")) for text in texts)
    )


def _looks_like_incomplete_inline_piece(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"(?:\bln|\blog|\bexp|\bsin|\bcos|\btan|[({\[+\-*/=])\s*$", stripped):
        return True
    return stripped.count("(") > stripped.count(")")


def raw_lines_to_math_aware_blocks(
    lines: list[RawLine],
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> list[RawBlock]:
    page_sizes = page_sizes or {}
    result: list[RawBlock] = []
    sorted_lines = _sort_lines_column_aware(lines, page_sizes)

    i = 0
    while i < len(sorted_lines):
        line = sorted_lines[i]

        if not looks_like_math_fragment(line):
            result.append(_raw_line_to_block(line, "line"))
            i += 1
            continue

        group = [line]
        j = i + 1
        while j < len(sorted_lines):
            nxt = sorted_lines[j]
            if not looks_like_math_fragment(nxt):
                if _looks_like_equation_number(nxt.text) and _can_group_equation_number(group, nxt):
                    group.append(nxt)
                    j += 1
                    continue
                break
            page_width = page_sizes.get(line.page, (0.0, 0.0))[0]
            if not _can_join_display_math_group(nxt, page_width=page_width):
                break
            if not _can_group_with_math_group(group, nxt):
                break
            group.append(nxt)
            j += 1

        page_width = page_sizes.get(line.page, (0.0, 0.0))[0]
        mode = classify_math_group(group, page_width=page_width)

        if mode == "display":
            result.append(_math_group_to_block(group, "formula_display_candidate"))
        elif mode == "inline":
            # Inline math belongs to text flow. Keep it mergeable with surrounding prose.
            for item in group:
                result.append(_raw_line_to_block(item, "line_with_inline_math"))
        else:
            # Ambiguous math is preserved as text; later stages must not replace it by [formule].
            for item in group:
                result.append(_raw_line_to_block(item, "ambiguous_math_line"))

        i = max(j, i + 1)

    return result


def _sort_lines_column_aware(
    lines: list[RawLine],
    page_sizes: dict[int, tuple[float, float]],
) -> list[RawLine]:
    by_page: dict[int, list[RawLine]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)
    result: list[RawLine] = []
    for page in sorted(by_page):
        page_lines = by_page[page]
        page_width = page_sizes.get(page, (0.0, 0.0))[0]
        layout = detect_columns(page_lines, page_width=page_width)
        result.extend(order_page_blocks(page_lines, layout))  # type: ignore[arg-type]
    return result


def _can_group_with_math_group(group: list[RawLine], candidate: RawLine) -> bool:
    return any(can_group_math_lines(previous, candidate) for previous in group[-6:])


def _can_join_display_math_group(line: RawLine, page_width: float | None) -> bool:
    text = line.text.strip()
    if not text:
        return False
    if len(text) <= 24 and line_math_score(line) >= 1.2:
        return True
    return classify_math_group([line], page_width=page_width) == "display"


def _looks_like_equation_number(text: str) -> bool:
    return bool(re.fullmatch(r"\(?\d+(?:\.\d+){0,3}\)?", text.strip()))


def _can_group_equation_number(group: list[RawLine], candidate: RawLine) -> bool:
    if not group or not _looks_like_equation_number(candidate.text):
        return False
    if candidate.page != group[-1].page:
        return False
    if candidate.bbox.x0 <= max(line.bbox.x1 for line in group):
        return False
    for line in group[-6:]:
        vertical_overlap = min(line.bbox.y1, candidate.bbox.y1) - max(line.bbox.y0, candidate.bbox.y0)
        if vertical_overlap >= min(line.bbox.height, candidate.bbox.height) * 0.35:
            return True
    return False


def _raw_line_to_block(line: RawLine, block_type: str) -> RawBlock:
    return RawBlock(
        text=line.text,
        bbox=line.bbox,
        page=line.page,
        block_type=block_type,
        lines=[line],
    )


def _math_group_to_block(group: list[RawLine], block_type: str) -> RawBlock:
    ordered = _order_math_group_lines(group)
    text = "\n".join(line.text.strip() for line in ordered if line.text.strip())
    return RawBlock(
        text=text,
        bbox=union_bbox(group),
        page=group[0].page,
        block_type=block_type,
        lines=group,
    )


def _order_math_group_lines(group: list[RawLine]) -> list[RawLine]:
    if len(group) <= 1:
        return group
    bbox = union_bbox(group)
    max_height = max((line.bbox.height for line in group), default=0.0)
    if max_height > 0 and bbox.height <= max_height * 2.2:
        return sorted(group, key=lambda line: line.bbox.x0)
    return sorted(group, key=lambda line: (line.bbox.y0, line.bbox.x0))
