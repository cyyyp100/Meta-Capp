from __future__ import annotations

import re
from statistics import median

from document.models import DocumentBlock, RawBlock


HEADING_KEYWORDS = (
    "abstract",
    "résumé",
    "resume",
    "keywords",
    "mots-clés",
    "mots cles",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "materials",
    "experiments",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
    "chapitre",
    "chapter",
    "section",
    "partie",
    "part",
    "appendix",
    "annexe",
    "définition",
    "definition",
    "déf.",
    "def.",
    "théorème",
    "theorem",
    "propriété",
    "property",
    "proposition",
    "preuve",
    "proof",
    "lemme",
    "lemma",
    "corollaire",
    "corollary",
    "exemple",
    "example",
    "remarque",
    "remark",
    "note",
    "attention",
    "warning",
    "exercice",
    "exercise",
    "question",
    "à retenir",
    "a retenir",
)

NUMBERING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+){0,4}\.?)|(?:[A-Z]\.)?\d+(?:\.\d+){0,4}\.?|[A-Z](?:\.\d+){0,4}\.?|[IVXLC]+[\).])(?:\s+|(?=[A-Za-zÀ-ÿ])).+",
    re.I,
)
NESTED_SECTION_RE = re.compile(r"^\s*(?:\d+|[A-Z])\.\s*\d+\.?(?:\s+|(?=[A-Za-zÀ-ÿ])).+", re.I)
CAPTION_RE = re.compile(
    r"^(Figure|Fig\.?|Schema|Schéma|Graphique|Diagramme|Illustration|Tableau|Table)\s*\d*[A-Za-z]?\s*[\.:]?\b",
    re.I,
)
IN_TEXT_FIGURE_REFERENCE_RE = re.compile(
    r"^(?:Figure|Fig\.?|Schema|Schéma|Graphique|Diagramme|Illustration|Tableau|Table)"
    r"\s*\d+[A-Za-z]?\s+"
    r"(?:depicts?|shows?|showcases?|illustrates?|presents?|describes?|reports?|"
    r"summari[sz]es?|compares?|includes?|one\s+can\s+see)\b",
    re.I,
)
CALLOUT_HEADING_RE = re.compile(r"^(?:à|a)\s+retenir$", re.I)
SIDE_METADATA_RE = re.compile(r"^(?:arxiv:|doi:|https?://|www\.|isbn\b|issn\b)", re.I)
EXACT_HEADING_LABEL_RE = re.compile(
    r"^(?:abstract|résumé|resume|keywords?|mots[- ]clés|mots[- ]cles)$",
    re.I,
)
INLINE_METADATA_RE = re.compile(r"\b(?:doi\.org|https?\s*:|www\.|arxiv:|isbn\b|issn\b)\b", re.I)
EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")
AFFILIATION_RE = re.compile(
    r"\b(?:university|college|institute|department|school|laborator(?:y|ies)|lab|"
    r"faculty|academy|hospital|centre|center|cnrs|inria)\b",
    re.I,
)
AUTHOR_MARKER_RE = re.compile(r"(?:[*†‡]|\\dagger|\\ddagger|\^\{?\d)")
CITATION_LINE_RE = re.compile(
    r"\bet\s+al\.?(?=\W|$)|\b[A-Z][A-Za-zÀ-ÿ'’-]+\s*&\s*[A-Z][A-Za-zÀ-ÿ'’-]+\b|"
    r"\(\s*[A-Z][A-Za-zÀ-ÿ'’-]+(?:\s+et\s+al\.?|\s*&\s*[A-Z][A-Za-zÀ-ÿ'’-]+)?\s*,\s*\d{4}",
    re.I,
)
SECTION_CROSS_REFERENCE_RE = re.compile(
    r"^\s*(?:section|sec\.?)\s+\d+(?:\.\d+)*\s+"
    r"(?:describes?|discuss(?:es)?|includes?|presents?|shows?|provides?|details?|introduces?|"
    r"explains?|contains?|is|are)\b",
    re.I,
)
DISCOURSE_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)+|[A-Z](?:\.\d+)+|[A-Z]\.)\.?\s+"
    r"(?:First|Second|Third|Finally|However|Moreover|Furthermore|This|These|The|We|Our|In|For|As)\b",
    re.I,
)
SEMANTIC_CALLOUT_RE = re.compile(
    r"^(?P<label>définition|definition|déf\.|def\.|théorème|theorem|propriété|property|"
    r"proposition|lemme|lemma|corollaire|corollary|exemple|example|remarque|remark|"
    r"attention|warning|exercice|exercise)(?P<sep>[\s:\.]+)(?P<body>.+)$",
    re.I,
)
SEMANTIC_TYPES = {
    "définition": "definition",
    "definition": "definition",
    "déf.": "definition",
    "def.": "definition",
    "théorème": "theorem",
    "theorem": "theorem",
    "propriété": "theorem",
    "property": "theorem",
    "proposition": "theorem",
    "lemme": "theorem",
    "lemma": "theorem",
    "corollaire": "theorem",
    "corollary": "theorem",
    "exemple": "example",
    "example": "example",
    "remarque": "remark",
    "remark": "remark",
    "attention": "warning",
    "warning": "warning",
    "exercice": "exercise",
    "exercise": "exercise",
}


def classify_blocks(raw_blocks: list[RawBlock]) -> list[DocumentBlock]:
    raw_blocks = merge_same_line_heading_fragments(raw_blocks)
    body_size = estimate_body_font_size(raw_blocks)
    classified: list[DocumentBlock] = []
    for raw in raw_blocks:
        block = classify_block(raw, body_size)
        for item in _split_embedded_heading_body(block):
            if item.text.strip() or item.type in {"figure", "table"}:
                classified.append(item)
    classified = _merge_numbered_heading_fragments(classified)
    return deduplicate_heading_blocks(_mark_caption_continuations(classified))


def _merge_numbered_heading_fragments(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        if i + 1 >= len(blocks):
            result.append(current)
            break
        nxt = blocks[i + 1]
        if _should_merge_numbered_heading_blocks(current, nxt):
            result.append(
                DocumentBlock(
                    type="heading",
                    text=f"{current.text.strip()} {nxt.text.strip()}",
                    page=current.page,
                    bbox=current.bbox.union(nxt.bbox) if current.bbox and nxt.bbox else current.bbox or nxt.bbox,
                    level=_heading_level_from_text(f"{current.text.strip()} {nxt.text.strip()}"),
                    confidence=max(float(current.confidence or 0.0), float(nxt.confidence or 0.0), 0.86),
                    metadata={**current.metadata, **nxt.metadata, "merged_numbered_heading_fragment": True},
                )
            )
            i += 2
            continue
        result.append(current)
        i += 1
    return result


def _should_merge_numbered_heading_blocks(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.page != right.page or left.bbox is None or right.bbox is None:
        return False
    if left.type not in {"paragraph", "heading"} or right.type != "heading":
        return False
    if not re.fullmatch(r"\d+(?:\.\d+)+\.?", (left.text or "").strip()):
        return False
    right_text = (right.text or "").strip()
    if not right_text or len(right_text.split()) > 10:
        return False
    vertical_overlap = min(left.bbox.y1, right.bbox.y1) - max(left.bbox.y0, right.bbox.y0)
    if vertical_overlap < min(left.bbox.height, right.bbox.height) * 0.55:
        return False
    gap = right.bbox.x0 - left.bbox.x1
    return 0 <= gap <= 64.0


def merge_same_line_heading_fragments(raw_blocks: list[RawBlock]) -> list[RawBlock]:
    if not raw_blocks:
        return []
    body_size = estimate_body_font_size(raw_blocks)
    result: list[RawBlock] = []
    i = 0
    while i < len(raw_blocks):
        current = raw_blocks[i]
        if i + 1 >= len(raw_blocks):
            result.append(current)
            break
        nxt = raw_blocks[i + 1]
        if _should_merge_heading_fragment(current, nxt, body_size):
            result.append(_merge_raw_blocks(current, nxt))
            i += 2
            continue
        result.append(current)
        i += 1
    return result


def _should_merge_heading_fragment(left: RawBlock, right: RawBlock, body_size: float) -> bool:
    if left.page != right.page or left.bbox is None or right.bbox is None:
        return False
    if left.block_type.startswith("formula") or right.block_type.startswith("formula"):
        return False
    vertical_overlap = min(left.bbox.y1, right.bbox.y1) - max(left.bbox.y0, right.bbox.y0)
    if vertical_overlap < min(left.bbox.height, right.bbox.height) * 0.65:
        return False
    gap = right.bbox.x0 - left.bbox.x1
    if gap < 0 or gap > max(40.0, body_size * 3.5):
        return False
    left_text = left.text.strip()
    right_text = right.text.strip()
    if not left_text or not right_text:
        return False
    if not re.fullmatch(r"\d+(?:\.\d+)+\.?", left_text):
        return False
    left_size = left.font_size or body_size
    right_size = right.font_size or body_size
    if right_text.isupper() and len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", right_text)) <= 8:
        return True
    return (
        (left.is_bold and right.is_bold)
        or (left_size >= body_size * 1.05 and right_size >= body_size * 1.05)
    )


def _merge_raw_blocks(left: RawBlock, right: RawBlock) -> RawBlock:
    return RawBlock(
        text=f"{left.text.strip()} {right.text.strip()}",
        bbox=left.bbox.union(right.bbox),
        page=left.page,
        block_type=right.block_type if right.block_type != "line" else left.block_type,
        lines=[*left.lines, *right.lines],
    )


def estimate_body_font_size(raw_blocks: list[RawBlock]) -> float:
    sizes = [
        line.font_size
        for block in raw_blocks
        for line in block.lines
        if line.font_size and line.text.strip()
    ]
    return float(median(sizes)) if sizes else 11.0


def classify_block(raw: RawBlock, body_size: float) -> DocumentBlock:
    text = re.sub(r"\s+", " ", raw.text).strip()
    metadata = {
        "raw_block_type": raw.block_type,
        "font_size": raw.font_size,
        "font_name": _dominant_font(raw),
        "is_bold": raw.is_bold,
    }

    if raw.block_type in {"formula_display_candidate", "formula_candidate"}:
        return DocumentBlock(
            type="formula",
            text=raw.text,
            page=raw.page,
            bbox=raw.bbox,
            confidence=0.78,
            metadata={
                **metadata,
                "source": "math_zone_detector",
                "formula_mode": "display",
                "render_mode": "pdf_crop",
                "preserve_bbox": True,
            },
        )

    if raw.block_type == "line_with_inline_math":
        metadata["contains_inline_math"] = True
        metadata["formula_mode"] = "inline"

    if raw.block_type == "ambiguous_math_line":
        metadata["contains_inline_math"] = True
        metadata["formula_mode"] = "ambiguous"

    if raw.block_type == "image":
        return DocumentBlock(
            type="figure",
            text=text,
            page=raw.page,
            bbox=raw.bbox,
            image_path=metadata.get("image_path"),
            confidence=0.8,
            metadata=metadata,
        )

    if (
        _looks_like_side_metadata(raw, text, body_size)
        or _looks_like_inline_metadata(text)
        or _looks_like_author_metadata(text, page=raw.page, bbox=raw.bbox, font_size=raw.font_size, body_size=body_size)
    ):
        metadata["is_metadata"] = True
        return DocumentBlock(
            type="paragraph",
            text=text,
            page=raw.page,
            bbox=raw.bbox,
            confidence=0.7,
            metadata=metadata,
        )

    semantic_type = _semantic_callout_type(text)
    if semantic_type is not None:
        metadata["detected_as"] = semantic_type
        return DocumentBlock(
            type=semantic_type,
            text=text,
            page=raw.page,
            bbox=raw.bbox,
            confidence=0.88,
            metadata=metadata,
        )

    if re.match(r"^\s*abstract\s*:\s*\S.+", text, re.I):
        metadata["detected_as"] = "abstract"
        return DocumentBlock(
            type="abstract",
            text=re.sub(r"^\s*abstract\s*:\s*", "", text, flags=re.I).strip(),
            page=raw.page,
            bbox=raw.bbox,
            confidence=0.88,
            metadata=metadata,
        )

    if _is_caption_text(text):
        metadata["is_caption"] = True
        return DocumentBlock(
            type="paragraph",
            text=text,
            page=raw.page,
            bbox=raw.bbox,
            confidence=0.78,
            metadata=metadata,
        )

    score = heading_score(raw, body_size)
    if score >= 3.0:
        return DocumentBlock(
            type="heading",
            text=text,
            page=raw.page,
            bbox=raw.bbox,
            level=heading_level(raw, body_size),
            confidence=min(1.0, 0.55 + score / 8.0),
            metadata=metadata,
        )

    return DocumentBlock(
        type="paragraph",
        text=text,
        page=raw.page,
        bbox=raw.bbox,
        confidence=0.9,
        metadata=metadata,
    )


def _mark_caption_continuations(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    previous_caption: DocumentBlock | None = None
    for block in blocks:
        if block.metadata.get("is_caption"):
            previous_caption = block
            result.append(block)
            continue

        if previous_caption is not None and _looks_like_caption_continuation(previous_caption, block):
            block.metadata["is_caption"] = True
            block.confidence = min(float(block.confidence or 1.0), 0.78)
            previous_caption = block
            result.append(block)
            continue

        previous_caption = None
        result.append(block)
    return result


def _looks_like_caption_continuation(previous: DocumentBlock, block: DocumentBlock) -> bool:
    if block.type != "paragraph" or previous.page != block.page:
        return False
    if previous.bbox is None or block.bbox is None:
        return False
    if block.metadata.get("is_metadata") or block.metadata.get("contains_inline_math"):
        return False

    gap = block.bbox.y0 - previous.bbox.y1
    previous_size = float((previous.metadata or {}).get("font_size") or 0.0)
    block_size = float((block.metadata or {}).get("font_size") or previous_size or 0.0)
    line_gap_limit = max(10.0, (previous_size or block_size or 9.0) * 1.35)
    if gap < -2.0 or gap > line_gap_limit:
        return False
    if previous_size and block_size > previous_size * 1.08:
        return False
    if abs(block.bbox.x0 - previous.bbox.x0) > 18.0:
        return False

    text = block.text.strip()
    if not text or _is_caption_text(text):
        return False
    if heading_score(
        RawBlock(
            text=text,
            bbox=block.bbox,
            page=int(block.page or 0),
            lines=[],
        ),
        body_size=max(previous_size, block_size, 9.0),
    ) >= 3.0:
        return False
    return True


def _is_caption_text(text: str) -> bool:
    return bool(CAPTION_RE.match(text) and not IN_TEXT_FIGURE_REFERENCE_RE.match(text))


def heading_score(raw: RawBlock, body_size: float) -> float:
    text = re.sub(r"\s+", " ", raw.text).strip()
    if not text or len(text) > 180:
        return 0.0
    if (
        _looks_like_side_metadata(raw, text, body_size)
        or _looks_like_inline_metadata(text)
        or _looks_like_author_metadata(text, page=raw.page, bbox=raw.bbox, font_size=raw.font_size, body_size=body_size)
        or _looks_like_sentence_not_heading(text)
    ):
        return 0.0

    score = 0.0
    size = raw.font_size or body_size
    if size > body_size * 1.18:
        score += 1.2
    if size > body_size * 1.35:
        score += 1.4
    if raw.is_bold:
        score += 0.9
    words = text.split()
    if len(words) <= 12:
        score += 0.7
    if len(words) <= 6:
        score += 0.4
    if NUMBERING_RE.match(text):
        score += 1.1
    if NESTED_SECTION_RE.match(text):
        score += 2.0
    if text.rstrip().endswith(":") and len(words) <= 10:
        score += 0.4
    lowered = text.casefold()
    if _starts_with_heading_keyword(lowered):
        score += 1.5
    # Boost pour les titres courts purement lexicaux ("Conclusion", "Chapter 7"…)
    # qui manquent souvent de signal gras/taille et resteraient sous le seuil sinon.
    if len(words) <= 3 and _starts_with_heading_keyword(lowered):
        score += 1.5
    stripped_numbering = _strip_numbering_prefix(lowered)
    if stripped_numbering != lowered and _starts_with_heading_keyword(stripped_numbering):
        score += 1.4
    if EXACT_HEADING_LABEL_RE.match(text):
        score += 2.2
    if re.match(r"^(?:appendix|annexe)\s+[A-Z0-9][\).]?\s+.+", lowered, re.I):
        score += 1.3
    if CALLOUT_HEADING_RE.match(text):
        score += 2.1
    if text.isupper() and len(text) > 3:
        score += 0.7
    if text.endswith(".") and len(words) > 8:
        score -= 0.8
    return score


def heading_level(raw: RawBlock, body_size: float) -> int:
    text = raw.text.strip()
    size = raw.font_size or body_size
    lowered = text.casefold()
    if lowered.startswith(("chapitre", "chapter", "partie", "part")) or size >= body_size * 1.55:
        return 1
    if lowered.startswith(("abstract", "résumé", "resume", "keywords", "mots-clés", "mots cles")):
        return 2
    if re.match(r"^\s*(?:appendix|annexe)\b", text, re.I):
        return 1
    if re.match(r"^\s*(?:\d+|[A-Z])\.\d+\.\d+", text, re.I):
        return 3
    if re.match(r"^\s*(?:\d+|[A-Z])\.\s+\d+\.", text, re.I) or re.match(
        r"^\s*(?:\d+|[A-Z])\.\d+",
        text,
        re.I,
    ):
        return 2
    if re.match(r"^\s*\d+\.?(?:\s+|(?=[A-Za-zÀ-ÿ]))\S+", text):
        return 1
    if re.match(r"^\s*[A-Z]\.(?:\s+|(?=[A-Za-zÀ-ÿ]))\S+", text, re.I):
        return 1
    if size >= body_size * 1.35:
        return 1
    return 2 if NUMBERING_RE.match(text) else 3


def _dominant_font(raw: RawBlock) -> str | None:
    names = [line.font_name for line in raw.lines if line.font_name]
    if not names:
        return None
    return max(set(names), key=names.count)


def _strip_numbering_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(?:(?:\d+(?:\.\d+){0,4}\.?)|(?:[a-z]\.)?\d+(?:\.\d+){0,4}\.?|[a-z](?:\.\d+){0,4}\.?|[ivxlc]+[\).])(?:\s+|(?=[A-Za-zÀ-ÿ]))",
        "",
        text,
        count=1,
        flags=re.I,
    )


def _starts_with_heading_keyword(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text or "").strip().casefold()
    for keyword in HEADING_KEYWORDS:
        key = keyword.casefold()
        if re.match(rf"^{re.escape(key)}(?:\b|[\s:.\-–—]|$)", clean, re.I):
            return True
    return False


def _looks_like_side_metadata(raw: RawBlock, text: str, body_size: float) -> bool:
    if not text:
        return False
    if SIDE_METADATA_RE.match(text.strip()):
        return True
    if raw.bbox is None:
        return False
    if raw.bbox.height <= 0 or raw.bbox.width <= 0:
        return False
    return raw.bbox.height > max(80.0, body_size * 6.0) and raw.bbox.width < max(42.0, body_size * 5.0)


def _looks_like_inline_metadata(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped:
        return False
    return bool(EMAIL_RE.search(stripped) or INLINE_METADATA_RE.search(stripped)) and len(stripped) <= 220


def _looks_like_author_metadata(
    text: str,
    page: int | None = None,
    bbox=None,
    font_size: float | None = None,
    body_size: float | None = None,
) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped:
        return False
    if CITATION_LINE_RE.search(stripped):
        return False
    if _looks_like_front_matter_name_or_location(stripped, page=page, bbox=bbox, font_size=font_size, body_size=body_size):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ0-9'’.-]+", stripped)
    if len(words) > 32:
        return False
    if page is not None and int(page or 0) > 2:
        return False
    if AUTHOR_MARKER_RE.search(stripped) and _looks_name_or_affiliation_like(stripped):
        return True
    if AFFILIATION_RE.search(stripped) and (page is None or int(page or 0) <= 2) and _looks_like_affiliation_line(stripped):
        return True
    if stripped.count(",") < 2:
        return False
    proper_words = re.findall(r"[A-Z][A-Za-zÀ-ÿ'’-]+", stripped)
    return len(proper_words) >= 4 and not re.search(r"[.!?]$", stripped)


def _looks_like_front_matter_name_or_location(
    text: str,
    *,
    page: int | None,
    bbox,
    font_size: float | None,
    body_size: float | None,
) -> bool:
    if page is None or int(page or 0) != 1:
        return False
    if bbox is None or bbox.y0 > 290.0:
        return False
    if font_size and body_size and font_size > body_size * 1.22:
        return False
    if _starts_with_heading_keyword(text.casefold()) or re.match(r"^\s*(?:\d+(?:\.\d+)*|[A-Z]\.)\s+", text):
        return False
    if re.search(r"[.!?;:]$", text):
        return False

    words = re.findall(r"[A-Za-zÀ-ÿ'’-]+", text)
    if not 1 <= len(words) <= 5:
        return False
    proper_words = re.findall(r"\b[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’-]{1,}\b", text)
    if text.count(",") == 1 and len(proper_words) >= 2:
        return True
    return len(words) >= 2 and len(proper_words) == len(words)


def _looks_name_or_affiliation_like(text: str) -> bool:
    proper_words = re.findall(r"\b[A-Z][A-Za-zÀ-ÿ'’.-]{1,}\b", text)
    return len(proper_words) >= 2 or bool(AFFILIATION_RE.search(text))


def _looks_like_affiliation_line(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"[.!?;:]\s*$", stripped):
        return False
    lowered = stripped.casefold()
    return not re.search(r"\b(?:we|this|these|our|method|model|result|results|figure|table)\b", lowered)


def _semantic_callout_type(text: str) -> str | None:
    match = SEMANTIC_CALLOUT_RE.match(re.sub(r"\s+", " ", text or "").strip())
    if not match:
        return None
    body = match.group("body").strip()
    label = match.group("label").casefold()
    sep = match.group("sep")
    # "attention"/"warning" are ambiguous English technical terms: require explicit punctuation
    if label in {"attention", "warning"} and not re.search(r"[:\.\!]", sep):
        return None
    if len(body.split()) < 4 and not re.search(r"[:.]", body) and not re.search(r"[:.]", sep):
        return None
    return SEMANTIC_TYPES.get(label)


def _looks_like_sentence_not_heading(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped:
        return False
    if SECTION_CROSS_REFERENCE_RE.match(stripped):
        return True
    if stripped.endswith("-") and len(stripped.split()) >= 6:
        return True
    if DISCOURSE_NUMBERED_HEADING_RE.match(stripped) and (len(stripped.split()) <= 2 or len(stripped.split()) >= 8):
        return True
    first = stripped[:1]
    if first.islower() and len(stripped.split()) >= 5:
        return True
    return False


def _split_embedded_heading_body(block: DocumentBlock) -> list[DocumentBlock]:
    callout_split = _split_embedded_callout_heading(block)
    if len(callout_split) != 1 or callout_split[0] is not block:
        return callout_split
    return _split_embedded_section_heading(block)


def _split_embedded_section_heading(block: DocumentBlock) -> list[DocumentBlock]:
    if block.type != "heading":
        return [block]
    text = re.sub(r"\s+", " ", block.text.strip())
    if len(text.split()) < 8:
        return [block]

    match = re.match(
        r"^(?P<title>(?:(?:\d+|[A-Z])\.)?(?:\d+(?:\.\d+)*|[A-Z])\.?\s+"
        r"[A-Z][A-Za-zÀ-ÿ0-9'’(),:/-]*(?:\s+(?:and|or|of|the|for|to|with|in|on|"
        r"[A-Z][A-Za-zÀ-ÿ0-9'’(),:/-]*)){0,6})\s+"
        r"(?P<body>(?:We|This|These|The|A|An|In|Here|Our|To|For|As)\b.+)$",
        text,
    )
    if not match:
        return [block]

    title = match.group("title").strip()
    body = match.group("body").strip()
    if len(title.split()) < 2 or len(body.split()) < 4:
        return [block]

    heading = DocumentBlock(
        type="heading",
        text=title,
        page=block.page,
        bbox=block.bbox,
        level=block.level,
        confidence=block.confidence,
        metadata={**block.metadata, "split_embedded_heading_body": True},
    )
    paragraph = DocumentBlock(
        type="paragraph",
        text=body,
        page=block.page,
        bbox=block.bbox,
        confidence=min(block.confidence, 0.86),
        metadata={**block.metadata, "split_embedded_heading_body": True},
    )
    return [heading, paragraph]


def _split_embedded_callout_heading(block: DocumentBlock) -> list[DocumentBlock]:
    if block.type != "paragraph":
        return [block]
    match = re.match(r"^(?P<body>.+?)\s+(?P<title>(?:À|A)\s+retenir)\s*$", block.text.strip(), re.I)
    if not match:
        return [block]

    body = match.group("body").strip()
    title = match.group("title").strip()
    if not body:
        return [block]

    body_block = DocumentBlock(
        type="paragraph",
        text=body,
        page=block.page,
        bbox=block.bbox,
        confidence=block.confidence,
        metadata={**block.metadata, "split_callout_heading": True},
    )
    heading_block = DocumentBlock(
        type="heading",
        text=title,
        page=block.page,
        bbox=block.bbox,
        level=3,
        confidence=block.confidence,
        metadata={**block.metadata, "split_callout_heading": True},
    )
    return [body_block, heading_block]


def deduplicate_heading_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Supprime les blocs heading dupliqués (même texte normalisé, même page)."""
    seen: set[tuple] = set()
    result: list[DocumentBlock] = []
    for block in blocks:
        if block.type == "heading":
            key = (block.page, re.sub(r"[\s.]+", " ", block.text or "").strip().casefold())
            if key in seen:
                continue
            seen.add(key)
        result.append(block)
    return result


def _heading_level_from_text(text: str) -> int:
    lowered = text.strip().casefold()
    if lowered.startswith(("chapitre", "chapter", "partie", "part", "appendix", "annexe")):
        return 1
    if lowered.startswith(("abstract", "résumé", "resume", "introduction", "conclusion",
                           "conclusions", "references", "acknowledgments", "acknowledgements")):
        return 2
    return 2


def promote_short_keyword_headings(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Reclassifie les blocs paragraph courts (≤3 mots) qui sont des titres lexicaux.

    Utilisé par l'extracteur OpenDataLoader qui se fie aux types de nœuds de la
    librairie et ne passe pas par heading_score().
    """
    result: list[DocumentBlock] = []
    for block in blocks:
        if block.type == "paragraph":
            text = re.sub(r"\s+", " ", block.text or "").strip()
            words = text.split()
            lowered = text.casefold()
            if len(words) <= 3 and any(lowered.startswith(kw) for kw in HEADING_KEYWORDS):
                block = DocumentBlock(
                    type="heading",
                    text=block.text,
                    page=block.page,
                    bbox=block.bbox,
                    level=_heading_level_from_text(text),
                    confidence=max(block.confidence, 0.7),
                    metadata={**block.metadata, "promoted_short_keyword_heading": True},
                )
        result.append(block)
    return result
