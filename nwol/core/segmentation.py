# core/segmentation.py — Segmentation en blocs sémantiques
from __future__ import annotations
import re
import logging
from typing import Any

logger = logging.getLogger("Segmentation")

# Légendes figure typiques
_FIGURE_CAPTION_RE = re.compile(
    r"^(Figure|Fig\.?|Schéma|Diagramme|Tableau|Table)\s+\d+", re.IGNORECASE
)
# Détection formule LaTeX simple
_FORMULA_INLINE_RE = re.compile(r"^\$.+\$$", re.DOTALL)
_FORMULA_DISPLAY_RE = re.compile(r"^\$\$.+\$\$$", re.DOTALL)
# Détection puces et listes numérotées
_LIST_RE = re.compile(r"^[•·▸►▶\-\*\+]\s|^\d+\.\s", re.UNICODE)
# Détection adresses e-mail
_EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.]+\b")
_METADATA_RE = re.compile(
    r"(@|doi\s*:|doi\.org|arxiv\s*:|issn\s*:|isbn\s*:|copyright|creative\s+commons|"
    r"\bgithub\b|\brepository\b|\bprojects?\s+(?:are\s+)?also\s+available\s+online\b)",
    re.IGNORECASE,
)
_AFFILIATION_RE = re.compile(
    r"\b(?:university|college|institute|department|school|laborator(?:y|ies)|lab|"
    r"faculty|academy|hospital|centre|center|cnrs|inria)\b",
    re.IGNORECASE,
)
_AUTHOR_MARKER_RE = re.compile(r"(?:[*†‡]|\\dagger|\\ddagger|\^\{?\d)")
_CODE_START_RE = re.compile(
    r"^\s*(?:def|class|for|while|if|elif|else|try|except|finally|with)\b.*:\s*(?:#.*)?$"
    r"|^\s*(?:import|from)\s+\w"
    r"|^\s*return\b\s+.+",
)


def segment_blocks(raw_blocks: list[dict]) -> list[dict]:
    """
    Prend les blocs bruts du parser et les normalise / enrichit :
    - détecte les légendes de figures
    - enrichit les blocs formules
    - tague les blocs code (police monospace heuristique)
    """
    result = []
    for b in raw_blocks:
        b = dict(b)  # copie
        btype = b.get("type", "paragraph")

        if btype == "caption":
            b["type"] = "paragraph"
            b["is_caption"] = True
            btype = "paragraph"

        if btype == "formula":
            text = b.get("text", "").strip()
            if text and not b.get("latex"):
                if _FORMULA_DISPLAY_RE.match(text):
                    b["latex"] = text[2:-2].strip()
                    b["display"] = True
                elif _FORMULA_INLINE_RE.match(text):
                    b["latex"] = text[1:-1].strip()
                    b["display"] = False

        elif btype == "paragraph":
            text = b.get("text", "").strip()
            if _FORMULA_DISPLAY_RE.match(text):
                b["type"] = "formula"
                b["latex"] = text[2:-2].strip()
                b["display"] = True
            elif _FORMULA_INLINE_RE.match(text):
                b["type"] = "formula"
                b["latex"] = text[1:-1].strip()
                b["display"] = False
            elif _FIGURE_CAPTION_RE.match(text):
                b["is_caption"] = True
            elif _looks_like_code(text):
                b["type"] = "code"
            elif text:
                b["text"] = _repair_corrupt_inline_dollar_text(text)

        result.append(b)

    result = _merge_list_items(result)
    result = _tag_metadata_blocks(result)
    result = _merge_short_text_continuations(result)
    result = _attach_visual_captions(result)
    before = len(result)
    result = [b for b in result if _block_is_reader_visible(b)]
    result = _drop_contained_formula_fragments(result)
    result = _drop_overlapping_semantic_formula_duplicates(result)
    result = _merge_acronym_formulas_into_text(result)
    result = _normalize_orphan_bullet_fragments(result)
    result = _drop_blocks_overlapping_figures(result)
    result = _drop_table_of_contents_pages(result)
    result = _drop_duplicate_semantic_headings(result)
    result = _drop_duplicate_heading_fragments(result)
    logger.debug(
        f"{len(result)} blocs segmentés "
        f"({_count_type(result, 'formula')} formules, "
        f"{_count_type(result, 'figure')} figures, "
        f"{_count_type(result, 'code')} code, "
        f"{before - len(result)} filtrés)"
    )
    return result


def _drop_duplicate_semantic_headings(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for index, block in enumerate(blocks):
        if block.get("type") in _HEADING_READER_TYPES and (block.get("metadata") or {}).get("semantic_only_block"):
            text = _normalized_heading_text(block.get("text") or "")
            page = block.get("page_number") or block.get("page_start") or block.get("page")
            for other in blocks[index + 1:index + 5]:
                if other.get("type") not in _HEADING_READER_TYPES:
                    continue
                other_page = other.get("page_number") or other.get("page_start") or other.get("page")
                if page is not None and other_page is not None and page != other_page:
                    continue
                other_text = _normalized_heading_text(other.get("text") or "")
                if other_text and text.startswith(other_text) and len(text) > len(other_text) + 8:
                    break
            else:
                result.append(block)
            continue
        result.append(block)
    return result


def _attach_visual_captions(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        if _is_caption_block(block):
            attached = _attach_caption_to_previous_visual(block, result)
            if attached:
                metadata = dict(block.get("metadata") or {})
                metadata["caption_attached_to_visual"] = True
                block = {**block, "metadata": metadata}
        result.append(block)
    return result


def _is_caption_block(block: dict) -> bool:
    if block.get("is_caption") or (block.get("metadata") or {}).get("is_caption"):
        return True
    if block.get("type") not in _TEXTUAL_READER_TYPES:
        return False
    return bool(_FIGURE_CAPTION_RE.match(str(block.get("text") or "").strip()))


def _attach_caption_to_previous_visual(caption_block: dict, previous_blocks: list[dict]) -> bool:
    text = str(caption_block.get("text") or "").strip()
    if not text:
        return False
    page = _block_page(caption_block)
    for visual in reversed(previous_blocks[-8:]):
        if visual.get("type") not in {"figure", "table"}:
            continue
        if _block_page(visual) != page:
            continue
        if not _visual_can_receive_caption(visual):
            continue
        if not _caption_is_geometrically_close(visual, caption_block):
            continue

        if not str(visual.get("caption") or "").strip():
            visual["caption"] = text
        metadata = dict(visual.get("metadata") or {})
        metadata["caption_display"] = True
        metadata.setdefault("caption_source", "reader_segmentation")
        visual["metadata"] = metadata
        return True
    return False


def _visual_can_receive_caption(block: dict) -> bool:
    metadata = block.get("metadata") or {}
    if block.get("type") == "figure":
        return bool(block.get("image_path") or metadata.get("context_asset_path"))
    if block.get("type") == "table":
        return bool(block.get("markdown") or block.get("html") or block.get("image_path") or metadata.get("table_image_path"))
    return False


def _caption_is_geometrically_close(visual: dict, caption: dict) -> bool:
    visual_bbox = visual.get("bbox")
    caption_bbox = caption.get("bbox")
    if not (
        isinstance(visual_bbox, (list, tuple))
        and len(visual_bbox) >= 4
        and isinstance(caption_bbox, (list, tuple))
        and len(caption_bbox) >= 4
    ):
        return True
    try:
        vx0, _vy0, vx1, vy1 = (float(value) for value in visual_bbox[:4])
        cx0, cy0, cx1, cy1 = (float(value) for value in caption_bbox[:4])
    except (TypeError, ValueError):
        return True
    vertical_gap = cy0 - vy1
    if vertical_gap < -8.0 or vertical_gap > 72.0:
        return False
    overlap = min(vx1, cx1) - max(vx0, cx0)
    caption_width = max(1.0, cx1 - cx0)
    visual_width = max(1.0, vx1 - vx0)
    return overlap >= min(caption_width, visual_width) * 0.35


def _drop_table_of_contents_pages(blocks: list[dict]) -> list[dict]:
    toc_pages = {_block_page(block) for block in blocks if _block_starts_table_of_contents(block)}
    toc_pages.discard(None)
    if not toc_pages:
        return blocks
    return [block for block in blocks if _block_page(block) not in toc_pages]


def _drop_contained_formula_fragments(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        if block.get("type") == "formula" and _formula_is_contained_fragment(block, result):
            continue
        result.append(block)
    return result


def _drop_overlapping_semantic_formula_duplicates(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for index, block in enumerate(blocks):
        if block.get("type") == "formula" and (block.get("metadata") or {}).get("semantic_only_block"):
            bbox = block.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                page = _block_page(block)
                duplicate = False
                for other in blocks[index + 1:index + 5]:
                    if other.get("type") != "formula" or _block_page(other) != page:
                        continue
                    other_bbox = other.get("bbox")
                    if not isinstance(other_bbox, (list, tuple)) or len(other_bbox) < 4:
                        continue
                    if _bbox_overlap_ratio(bbox, other_bbox) >= 0.55:
                        duplicate = True
                        break
                if duplicate:
                    continue
        result.append(block)
    return result


def _merge_acronym_formulas_into_text(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        if block.get("type") == "formula" and _merge_acronym_formula_with_previous_text(block, result):
            continue
        result.append(block)
    return result


def _merge_acronym_formula_with_previous_text(block: dict, previous_blocks: list[dict]) -> bool:
    acronym = _formula_acronym_text(block)
    if acronym is None:
        return False
    page = _block_page(block)
    for previous in reversed(previous_blocks[-3:]):
        if _block_page(previous) != page or previous.get("type") not in _TEXTUAL_READER_TYPES:
            continue
        previous_text = str(previous.get("text") or "")
        if re.search(r"MAML\s*\+?\s*\+", previous_text, re.I):
            return True
        previous["text"] = f"{previous_text.rstrip()} {acronym}.".strip()
        previous["bbox"] = _union_bbox_values(previous.get("bbox"), block.get("bbox"))
        return True
    return False


def _formula_acronym_text(block: dict) -> str | None:
    text = re.sub(r"[\s$]+", " ", str(block.get("text") or block.get("latex") or "")).strip(" .")
    if re.fullmatch(r"MAML\s*\+\s*\+", text, re.I):
        return "MAML++"
    return None


def _normalize_orphan_bullet_fragments(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        if block.get("type") != "bullet_list" or not _bullet_list_is_orphan_text_fragment(block):
            result.append(block)
            continue
        text = _orphan_bullet_text(block)
        if re.match(r"^denotes\b", text, re.I):
            continue
        result.append({**block, "type": "paragraph", "text": text, "items": []})
    return result


def _bullet_list_is_orphan_text_fragment(block: dict) -> bool:
    text = _orphan_bullet_text(block)
    if not text:
        return False
    return bool(re.match(r"^(?:that|where|denotes|which|when|while|because|and|or)\b", text, re.I))


def _orphan_bullet_text(block: dict) -> str:
    items = block.get("items") or []
    if len(items) == 1:
        raw = str(items[0] or "")
    else:
        raw = _block_text_for_filter(block)
    return re.sub(r"^[•·▸►▶\-\*\+]\s*", "", raw).strip()


def _drop_blocks_overlapping_figures(blocks: list[dict]) -> list[dict]:
    figures = [block for block in blocks if block.get("type") == "figure" and block.get("bbox")]
    if not figures:
        return blocks
    result: list[dict] = []
    for block in blocks:
        if block.get("type") != "figure" and _block_is_visual_residue(block, figures):
            continue
        result.append(block)
    return result


def _block_is_visual_residue(block: dict, figures: list[dict]) -> bool:
    if block.get("type") not in {"paragraph", "text", "table", "bullet_list"}:
        return False
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    page = _block_page(block)
    text = _block_text_for_filter(block)
    if not _looks_like_visual_residue_text(text, block.get("type")):
        return False
    for figure in figures:
        if _block_page(figure) != page:
            continue
        fig_bbox = figure.get("bbox")
        if not isinstance(fig_bbox, (list, tuple)) or len(fig_bbox) < 4:
            continue
        if _bbox_overlap_ratio(fig_bbox, bbox) >= 0.18 or _bbox_contains(fig_bbox, bbox, tolerance=10.0):
            return True
    return False


def _block_text_for_filter(block: dict) -> str:
    if block.get("type") == "bullet_list":
        return " ".join(str(item) for item in block.get("items") or [])
    return str(block.get("text") or block.get("caption") or "")


def _looks_like_visual_residue_text(text: str, block_type: str | None) -> bool:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if block_type == "table" and not clean:
        return True
    if block_type == "table" and re.search(r"\bfigure\s+\d+\b", clean, re.I):
        return True
    if block_type == "bullet_list" and len(clean) <= 180:
        return True
    if len(clean) <= 220 and re.search(r"\b(?:epoch|seed|accuracy|figure\s+\d+|maml\+\+)\b", clean, re.I):
        return True
    numeric_tokens = len(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", clean))
    return numeric_tokens >= 4 and len(clean) <= 180


def _formula_is_contained_fragment(block: dict, previous_blocks: list[dict]) -> bool:
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    text = re.sub(r"\s+", " ", str(block.get("text") or block.get("latex") or "")).strip()
    metadata = block.get("metadata") or {}
    if len(text) > 24 and not metadata.get("semantic_only_block"):
        return False
    page = _block_page(block)
    for previous in reversed(previous_blocks[-6:]):
        if previous.get("type") != "formula" or _block_page(previous) != page:
            continue
        prev_bbox = previous.get("bbox")
        if not isinstance(prev_bbox, (list, tuple)) or len(prev_bbox) < 4:
            continue
        if _bbox_contains(prev_bbox, bbox, tolerance=3.0):
            return True
    return False


def _bbox_contains(outer: list | tuple, inner: list | tuple, *, tolerance: float = 0.0) -> bool:
    try:
        return (
            float(outer[0]) - tolerance <= float(inner[0])
            and float(outer[1]) - tolerance <= float(inner[1])
            and float(outer[2]) + tolerance >= float(inner[2])
            and float(outer[3]) + tolerance >= float(inner[3])
        )
    except (TypeError, ValueError):
        return False


def _bbox_overlap_ratio(left: list | tuple, right: list | tuple) -> float:
    try:
        lx0, ly0, lx1, ly1 = (float(value) for value in left[:4])
        rx0, ry0, rx1, ry1 = (float(value) for value in right[:4])
    except (TypeError, ValueError):
        return 0.0
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    right_area = max(1.0, (rx1 - rx0) * (ry1 - ry0))
    return inter / right_area


def _block_starts_table_of_contents(block: dict) -> bool:
    if block.get("type") not in _HEADING_READER_TYPES:
        return False
    text = _normalized_heading_text(block.get("text") or "")
    return text in {"contents", "table of contents", "sommaire"}


def _drop_duplicate_heading_fragments(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        if (
            block.get("type") in _HEADING_READER_TYPES
            and result
            and _heading_fragment_repeats_previous(result[-1], block)
        ):
            continue
        result.append(block)
    return result


def _heading_fragment_repeats_previous(previous: dict, current: dict) -> bool:
    if previous.get("type") not in _HEADING_READER_TYPES:
        return False
    if _block_page(previous) != _block_page(current):
        return False
    current_text = _normalized_heading_text(current.get("text") or "")
    previous_text = _normalized_heading_text(previous.get("text") or "")
    if not current_text or not previous_text:
        return False

    prev_bbox = previous.get("bbox")
    curr_bbox = current.get("bbox")
    if not (
        isinstance(prev_bbox, (list, tuple))
        and len(prev_bbox) >= 4
        and isinstance(curr_bbox, (list, tuple))
        and len(curr_bbox) >= 4
    ):
        return False
    try:
        vertical_gap = float(curr_bbox[1]) - float(prev_bbox[3])
    except (TypeError, ValueError):
        return False
    if vertical_gap > 42.0:
        return False

    if current_text.startswith(previous_text) and len(current_text) > len(previous_text) + 8:
        return True
    if len(current_text.split()) > 4:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(current_text)}$", previous_text))


def _block_page(block: dict) -> int | None:
    page = block.get("page_number") or block.get("page_start") or block.get("page")
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_heading_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    text = re.sub(r"^(\d+(?:\.\d+)*)\.\s+", r"\1 ", text)
    return re.sub(r"[.:]\s*$", "", text)


def _merge_list_items(blocks: list[dict]) -> list[dict]:
    """Fusionne les blocs paragraph consécutifs qui sont des puces de liste."""
    result: list[dict] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.get("type") == "paragraph" and _LIST_RE.match(b.get("text", "").strip()):
            items = [b.get("text", "").strip()]
            j = i + 1
            while (
                j < len(blocks)
                and blocks[j].get("type") == "paragraph"
                and _LIST_RE.match(blocks[j].get("text", "").strip())
            ):
                items.append(blocks[j].get("text", "").strip())
                j += 1
            if len(items) > 1:
                result.append({**b, "text": "\n".join(items), "is_list": True})
                i = j
                continue
        result.append(b)
        i += 1
    return result


def _tag_metadata_blocks(blocks: list[dict]) -> list[dict]:
    """Marque les blocs de front matter pour qu'ils ne déclenchent pas de Q&A."""
    result = []
    for b in blocks:
        if b.get("type") in {"paragraph", "text", "heading"}:
            text = re.sub(r"\s+", " ", str(b.get("text", "") or "")).strip()
            if _looks_like_metadata_text(text):
                metadata = dict(b.get("metadata") or {})
                metadata["is_metadata"] = True
                b = {**b, "is_metadata": True, "metadata": metadata}
        result.append(b)
    return result


def _merge_short_text_continuations(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    for block in blocks:
        if result and _short_text_continues_previous(result[-1], block):
            previous = dict(result[-1])
            previous_text = str(previous.get("text") or "").rstrip()
            text = str(block.get("text") or "").strip()
            if text and _normalize_plain_text(text) not in _normalize_plain_text(previous_text):
                previous["text"] = f"{previous_text} {text}".strip()
                previous["bbox"] = _union_bbox_values(previous.get("bbox"), block.get("bbox"))
                result[-1] = previous
            continue
        result.append(block)
    return result


def _short_text_continues_previous(previous: dict, current: dict) -> bool:
    if previous.get("type") not in _TEXTUAL_READER_TYPES or current.get("type") not in _TEXTUAL_READER_TYPES:
        return False
    if _block_page(previous) != _block_page(current):
        return False
    text = str(current.get("text") or "").strip()
    if not text or len(text) > 80:
        return False
    metadata = current.get("metadata") or {}
    if metadata.get("is_caption") or current.get("is_caption") or metadata.get("is_header_footer"):
        return False
    prev_bbox = previous.get("bbox")
    curr_bbox = current.get("bbox")
    if not (
        isinstance(prev_bbox, (list, tuple))
        and len(prev_bbox) >= 4
        and isinstance(curr_bbox, (list, tuple))
        and len(curr_bbox) >= 4
    ):
        return False
    try:
        vertical_gap = float(curr_bbox[1]) - float(prev_bbox[3])
        x_delta = abs(float(curr_bbox[0]) - float(prev_bbox[0]))
    except (TypeError, ValueError):
        return False
    if vertical_gap > 10.0 or x_delta > 24.0:
        return False
    return bool(re.match(r"^[a-zà-ÿ0-9]", text) or re.fullmatch(r"[A-Za-zÀ-ÿ]+[.!?]?", text))


def _normalize_plain_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _union_bbox_values(left: Any, right: Any) -> list | Any:
    if not (
        isinstance(left, (list, tuple))
        and len(left) >= 4
        and isinstance(right, (list, tuple))
        and len(right) >= 4
    ):
        return left
    try:
        return [
            min(float(left[0]), float(right[0])),
            min(float(left[1]), float(right[1])),
            max(float(left[2]), float(right[2])),
            max(float(left[3]), float(right[3])),
        ]
    except (TypeError, ValueError):
        return left


def _looks_like_metadata_text(text: str) -> bool:
    if not text:
        return False
    if _EMAIL_RE.search(text) or _METADATA_RE.search(text):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ0-9'’.-]+", text)
    if len(words) > 32:
        return False
    if _AUTHOR_MARKER_RE.search(text) and _looks_name_or_affiliation_like(text):
        return True
    return bool(_AFFILIATION_RE.search(text) and _looks_like_affiliation_line(text))


def _looks_name_or_affiliation_like(text: str) -> bool:
    proper_words = re.findall(r"\b[A-Z][A-Za-zÀ-ÿ'’.-]{1,}\b", text)
    return len(proper_words) >= 2 or bool(_AFFILIATION_RE.search(text))


def _looks_like_affiliation_line(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"[.!?;:]\s*$", stripped):
        return False
    lowered = stripped.casefold()
    return not re.search(r"\b(?:we|this|these|our|method|model|result|results|figure|table)\b", lowered)


def _looks_like_code(text: str) -> bool:
    """Heuristique : lignes courtes avec indentation ou mots-clés typiques."""
    lines = text.splitlines()
    if not lines:
        return False
    stripped = text.lstrip()
    if stripped.startswith((">>> ", "$ ", "# ", "if __name__")):
        return True
    indented = sum(1 for l in lines if l.startswith(("    ", "\t")))
    has_keyword = any(_CODE_START_RE.match(line) for line in lines if line.strip())
    return has_keyword or (indented > len(lines) / 2)


def _repair_corrupt_inline_dollar_text(text: str) -> str:
    repaired = re.sub(r"(?<=[A-Za-zÀ-ÿ])\$(?=[A-Za-zÀ-ÿ])", "", text or "")
    repaired = re.sub(r"(?<=[A-Za-zÀ-ÿ])\$(?=\s*\\(?:rightarrow|leftarrow|to)\b)", "", repaired)
    repaired = re.sub(r"\$(?=\([A-Za-zÀ-ÿ])", "", repaired)
    repaired = re.sub(r"\$\s*(?=[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{3,})", "", repaired)
    repaired = re.sub(r"\$(?=[A-Za-z][_^])", "", repaired)
    repaired = re.sub(r"(\})\s*(?=(?:denotes|where|is|are)\b)", r"\1 ", repaired)
    return repaired


def _count_type(blocks: list[dict], btype: str) -> int:
    return sum(1 for b in blocks if b.get("type") == btype)


# ---------------------------------------------------------------------------
# Reader-visibility filter
# ---------------------------------------------------------------------------

_TEXTUAL_READER_TYPES = frozenset({
    "paragraph", "text", "abstract", "definition", "theorem",
    "example", "remark", "warning", "exercise", "question", "quote",
})
_HEADING_READER_TYPES = frozenset({"heading", "subheading", "subsubheading"})

# Short section heading words that are legitimate even without numbering.
_KNOWN_SHORT_HEADINGS = frozenset({
    "abstract", "introduction", "background", "related", "methods", "method",
    "experiments", "experiment", "results", "result", "discussion", "conclusion",
    "conclusions", "references", "reference", "appendix", "summary", "overview",
    "approach", "model", "training", "inference", "dataset", "evaluation",
    "notation", "baseline", "details", "proof", "lemma", "theorem", "definition",
    "example", "motivation", "analysis", "comparison", "notes", "setup",
    "future", "contributions", "contribution", "limitations", "limitation",
    "preliminaries", "background", "methodology", "related work", "acknowledgments",
    "acknowledgements", "chapter", "chapitre", "contents", "table of contents", "sommaire",
    "additional results",
})


def _block_is_reader_visible(b: dict[str, Any]) -> bool:
    """Return False for blocks that should not be shown in the reading pane."""
    metadata: dict[str, Any] = b.get("metadata") or {}

    # Keep front-matter metadata in cached blocks so downstream code can see why
    # it was skipped. The reader engine suppresses it at playback time.
    if (
        b.get("is_reference")
        or metadata.get("is_reference")
        or metadata.get("is_header_footer")
    ):
        return False

    btype = b.get("type") or "paragraph"
    text = (b.get("text") or "").strip()
    is_caption = bool(b.get("is_caption") or metadata.get("is_caption"))

    # Figures: keep only if they have an image or caption.
    if btype == "figure":
        return bool(b.get("image_path") or b.get("caption") or text)

    # Formulas: keep only if they have LaTeX or an image.
    if btype == "formula":
        return bool(b.get("latex") or b.get("image_path") or metadata.get("formula_image_path"))

    # Tables: keep only if they have content.
    if btype == "table":
        return bool(text or b.get("markdown") or b.get("html") or b.get("image_path") or metadata.get("table_image_path"))

    # Code blocks can be produced by the light segmentation pass. Respect the
    # pipeline visibility flag so short non-displayable fragments cannot be
    # revived by a later type change.
    if btype == "code":
        return bool(text) and metadata.get("displayable") is not False

    # Heading types: filter figure-legend labels masquerading as headings.
    if btype in _HEADING_READER_TYPES:
        if not text:
            return False
        return not _heading_looks_like_figure_label(b, text)

    # Textual types: the pipeline marks them displayable=False when they are
    # too short, are column fragments, or contain no substantive content.
    if btype in _TEXTUAL_READER_TYPES:
        if not text:
            return False
        # Captions should be rendered through their associated visual block.
        # If no visual/table was associated, do not leave an orphan legend in
        # the reading flow.
        if is_caption:
            if metadata.get("caption_attached_to_visual"):
                return False
            if metadata.get("visual_assets") or metadata.get("llm_assets"):
                return True
            return False
        if _looks_like_front_matter_link_text(text):
            return False
        # displayable=False set by the document pipeline → skip.
        if metadata.get("displayable") is False:
            return False
        return True

    # Bullet lists: keep if they have items.
    if btype == "bullet_list":
        if metadata.get("displayable") is False:
            return False
        return bool(b.get("items"))

    # Default: keep if there is any text content.
    return bool(text)


def _looks_like_front_matter_link_text(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) > 140:
        return False
    return bool(
        re.search(r"\bgithub\b|\brepository\b", clean, re.I)
        or re.search(r"\bprojects?\s+(?:are\s+)?also\s+available\s+online\b", clean, re.I)
    )


def _heading_looks_like_figure_label(block: dict[str, Any], text: str) -> bool:
    """True if a heading block appears to be a figure/diagram label, not a section title."""
    text = text.strip()
    if _looks_like_reference_heading(text):
        return True
    if not text or len(text) > 50:
        return False
    metadata = block.get("metadata") or {}
    bbox = block.get("bbox")
    try:
        font_size = float(metadata.get("font_size") or 0.0)
        y0 = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 2 else 9999.0
        page_height = float(metadata.get("page_height") or 0.0)
    except (TypeError, ValueError):
        font_size = 0.0
        y0 = 9999.0
        page_height = 0.0
    if font_size >= 13.0 and page_height > 0.0 and y0 <= page_height * 0.35:
        return False
    # Has section numbering (e.g. "3.2. …") → real heading.
    if re.match(r"^\d", text):
        return False
    if re.match(r"^(?:chapter|chapitre)\s+\d+\b", text, re.I):
        return False
    # Long headings with multiple meaningful words → real heading.
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)
    if len(words) >= 4:
        return False
    # Known section keywords → real heading even if short.
    if text.casefold() in _KNOWN_SHORT_HEADINGS:
        return False
    # Multi-word headings that look like natural section titles (≥ 3 words total).
    all_words = text.split()
    if len(all_words) >= 3 and len(text) >= 18:
        return False
    # Anything remaining that is short with no section structure is a figure label.
    return len(text) <= 30


def _looks_like_reference_heading(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if len(stripped) < 40:
        return False
    if not re.search(r"\b(?:19|20)\d{2}\b", stripped):
        return False
    if re.search(r"\b(?:arxiv|preprint|proceedings|conference|journal|methodology|probability|pp\.)\b", stripped, re.I):
        return True
    return bool(re.search(r",\s*\d+\s*\(\d+\)\s*:\s*\d+", stripped))
