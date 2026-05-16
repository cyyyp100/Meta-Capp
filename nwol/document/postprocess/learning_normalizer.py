from __future__ import annotations

import re

from document.models import DocumentBlock

SEMANTIC_PREFIXES: dict[str, str] = {
    "abstract": "abstract",
    "résumé": "abstract",
    "resume": "abstract",
    "définition": "definition",
    "definition": "definition",
    "def.": "definition",
    "déf.": "definition",
    "exemple": "example",
    "example": "example",
    "ex.": "example",
    "remarque": "remark",
    "remark": "remark",
    "note": "remark",
    "attention": "warning",
    "warning": "warning",
    "avertissement": "warning",
    "théorème": "theorem",
    "theorem": "theorem",
    "théo.": "theorem",
    "propriété": "theorem",
    "propriete": "theorem",
    "property": "theorem",
    "proposition": "theorem",
    "lemme": "theorem",
    "lemma": "theorem",
    "corollaire": "theorem",
    "corollary": "theorem",
    "question": "question",
    "exercice": "exercise",
    "exercise": "exercise",
    "exo.": "exercise",
}

_MATH_CHARS = frozenset("=+-*/∑∫≤≥∈∉Δαβγλθπφψω()[]{}^_\\|<>")
_BIBLIO_MARKER_RE = re.compile(
    r"^(?:\[?(?:CrossRef|PubMed|Google Scholar)\]?\s*)+$",
    re.I,
)
_SECTION_HEADING_RE = re.compile(
    r"^\d+(?:\.\d+)*\.?(?:\s+|(?=[A-Za-zÀ-ÿ]))[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’(),:;/\-\s]{2,100}$"
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+|[A-Z](?:\.\d+)+|[A-Z]\.)(?:\.)?(?:\s+|(?=[A-Za-zÀ-ÿ]))"
    r"(?P<label>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’(),:;/\-\s]{2,130})$",
    re.I,
)
_EMBEDDED_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+|[A-Z](?:\.\d+)+|[A-Z]\.)(?:\.)?(?:\s+|(?=[A-Za-zÀ-ÿ]))"
    r"(?P<label>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9’’(),:;/\-\s]{2,110}?)\s+"
    r"(?P<body>(?:We|This|These|The|A|An|In|Here|Our|To|For|As|However|Although|Given|It|They|"
    r"From|Nous|Cette|Ces|Le|La|Les|Dans|Pour|Ainsi|Cependant)\b.+|[A-Z][a-z]+:\s.+)$",
    re.I,
)
# Detects a section number that appears mid-paragraph (after a sentence boundary).
_MID_PARA_SECTION_RE = re.compile(
    r"(?<=[.!?])\s+(\d+(?:\.\d+)+\.?\s+[A-Z][A-Za-zÀ-ÿ])",
)
_DISCOURSE_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)+|[A-Z](?:\.\d+)+|[A-Z]\.)\.?\s+"
    r"(?:First|Second|Third|Finally|However|Moreover|Furthermore|This|These|The|We|Our|In|For|As)\b",
    re.I,
)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[•▪◦]|\-|\*|\d+[\).]|[A-Za-z][\).])\s+")
_SHORT_TEXT_LABEL_RE = re.compile(r"^[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'’(),.:;\-\s/%]+$")


def normalize_for_learning(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    blocks = split_overextended_numbered_headings(blocks)
    blocks = split_mid_paragraph_section_headings(blocks)
    blocks = promote_numbered_paragraph_headings(blocks)
    blocks = assign_stable_ids(blocks)
    blocks = fix_wrong_formula_blocks(blocks)
    blocks = detect_semantic_callouts(blocks)
    blocks = merge_semantic_heading_callouts(blocks)
    blocks = compute_block_confidence(blocks)
    return blocks


def split_mid_paragraph_section_headings(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Split paragraphs that have a numbered section heading embedded mid-text.

    Example: "...respectively. 3.1 Encoder and Decoder Stacks Encoder: The encoder..."
    becomes three blocks: paragraph / heading / paragraph.
    """
    result: list[DocumentBlock] = []
    for block in blocks:
        if block.type not in {"paragraph", "text"} or not block.text:
            result.append(block)
            continue

        text = block.text.strip()
        match = _MID_PARA_SECTION_RE.search(text)
        if not match:
            result.append(block)
            continue

        # Position right after the sentence-ending punctuation
        sentence_end_pos = match.start() + 1  # after the [.!?]
        suffix_start = match.start(1)          # start of "3.1 ..."

        pre_text = text[:sentence_end_pos].strip()
        suffix = text[suffix_start:].strip()

        if not pre_text or not suffix:
            result.append(block)
            continue

        # Try to detect heading+body within the suffix
        suffix_block = DocumentBlock(
            type=block.type,
            text=suffix,
            page=block.page,
            bbox=block.bbox,
            confidence=block.confidence,
            metadata=dict(block.metadata),
        )
        split = _split_numbered_heading_body(suffix_block, suffix)
        if split is None:
            # Suffix might be heading-only (no body): promote if it looks like a heading
            if _looks_like_numbered_paragraph_heading(suffix):
                pre_block = DocumentBlock(
                    type="paragraph", text=pre_text, page=block.page,
                    bbox=block.bbox, confidence=block.confidence,
                    metadata=dict(block.metadata),
                )
                heading_block = DocumentBlock(
                    type="heading", text=_normalize_numbered_heading_text(suffix) or suffix,
                    page=block.page, bbox=block.bbox,
                    level=_heading_level_from_numbering(suffix),
                    confidence=min(block.confidence, 0.86),
                    metadata={**block.metadata, "detected_as": "heading", "promoted_from": block.type},
                )
                result.extend([pre_block, heading_block])
            else:
                result.append(block)
            continue

        pre_block = DocumentBlock(
            type="paragraph", text=pre_text, page=block.page,
            bbox=block.bbox, confidence=block.confidence,
            metadata=dict(block.metadata),
        )
        result.append(pre_block)
        result.extend(split)

    return result


def split_overextended_numbered_headings(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for block in blocks:
        if block.type != "heading":
            result.append(block)
            continue

        split = _split_numbered_heading_body(block, str(block.text or ""))
        if split is None:
            result.append(block)
            continue
        result.extend(split)
    return result


def promote_numbered_paragraph_headings(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    for block in blocks:
        if block.type not in {"paragraph", "text", "bullet_list"}:
            result.append(block)
            continue

        text = _numbered_heading_source_text(block)
        if not text:
            result.append(block)
            continue

        split = _split_numbered_heading_body(block, text)
        if split is not None:
            result.extend(split)
            continue

        if _looks_like_numbered_paragraph_heading(text):
            original_type = block.type
            block.type = "heading"
            block.text = _normalize_numbered_heading_text(text) or text
            block.level = _heading_level_from_numbering(text)
            block.items = None
            block.metadata.setdefault("detected_as", "heading")
            block.metadata.setdefault("promoted_from", original_type)
        result.append(block)
    return result


def _numbered_heading_source_text(block: DocumentBlock) -> str:
    if block.type == "bullet_list" and block.items:
        text = str(block.items[0] or "")
    else:
        text = str(block.text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = _BULLET_PREFIX_RE.sub("", text, count=1).strip()
    return text


def assign_stable_ids(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    for index, block in enumerate(blocks):
        if not block.id:
            page = block.page or 0
            block.id = f"p{page}_b{index}"
    return blocks


def fix_wrong_formula_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    for block in blocks:
        if block.type != "formula":
            continue
        text = (block.text or block.latex or "").strip()
        clean_text = _strip_formula_delimiters(text)
        if _looks_like_numbered_heading(text):
            block.type = "heading"
            block.text = clean_text
            block.level = _heading_level_from_numbering(text)
            block.latex = None
            block.metadata.setdefault("corrected_from", "formula")
            block.metadata.setdefault("detected_as", "heading")
            continue
        if _looks_like_plain_text_label(text):
            block.type = "paragraph"
            block.text = clean_text
            block.latex = None
            block.metadata.setdefault("corrected_from", "formula")
            continue
        if _looks_like_natural_text(text):
            block.type = "paragraph"
            block.text = clean_text
            block.latex = None
            block.metadata.setdefault("corrected_from", "formula")
    return blocks


def detect_semantic_callouts(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    for block in blocks:
        if block.type not in {"paragraph", "text", "heading"}:
            continue
        text = (block.text or "").strip()
        if not text:
            continue
        text_lower = text.lower()
        for prefix, semantic_type in SEMANTIC_PREFIXES.items():
            if block.type == "heading" and _semantic_heading_label_only(text_lower, prefix):
                continue
            if _semantic_prefix_matches(text_lower, prefix):
                block.type = semantic_type
                block.level = None
                block.metadata.setdefault("detected_as", semantic_type)
                break
    return blocks


def _semantic_prefix_matches(text_lower: str, prefix: str) -> bool:
    escaped = re.escape(prefix)
    return bool(
        text_lower.startswith(prefix + ":")
        or text_lower.startswith(prefix + " :")
        or text_lower.startswith(prefix + "\n")
        or text_lower.startswith(prefix + ".")
        or re.match(rf"^{escaped}\s+\d+(?:\.\d+)*", text_lower)
    )


def _semantic_heading_label_only(text_lower: str, prefix: str) -> bool:
    escaped = re.escape(prefix)
    return bool(re.fullmatch(rf"{escaped}(?:\s+\d+(?:\.\d+)*)?\s*[.:]?", text_lower.strip()))


def merge_semantic_heading_callouts(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        semantic_type = _semantic_heading_type(block)
        if semantic_type is None:
            result.append(block)
            i += 1
            continue

        if i + 1 < len(blocks) and _can_merge_callout_body(block, blocks[i + 1]):
            body = blocks[i + 1]
            text = " ".join(part for part in (block.text.strip(), body.text.strip()) if part)
            bbox = block.bbox.union(body.bbox) if block.bbox and body.bbox else block.bbox or body.bbox
            result.append(
                DocumentBlock(
                    type=semantic_type,
                    text=text,
                    page=block.page,
                    bbox=bbox,
                    confidence=min(block.confidence, body.confidence),
                    metadata={
                        **block.metadata,
                        "detected_as": semantic_type,
                        "merged_callout_heading": True,
                    },
                )
            )
            i += 2
            continue

        block.type = semantic_type
        block.level = None
        block.metadata.setdefault("detected_as", semantic_type)
        result.append(block)
        i += 1
    return result


def compute_block_confidence(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    for block in blocks:
        confidence = block.confidence
        text = (block.text or "").strip()

        if not text and block.type not in {"figure", "formula"}:
            confidence = min(confidence, 0.40)
        if block.metadata.get("corrected_from"):
            confidence = min(confidence, 0.85)
        if block.metadata.get("detected_as"):
            confidence = min(confidence, 0.90)
        if block.type == "paragraph" and 0 < len(text) < 20:
            confidence = min(confidence, 0.60)
        if block.type == "formula" and not block.latex:
            confidence = min(confidence, 0.50)

        block.confidence = round(max(0.0, min(1.0, confidence)), 3)
    return blocks


def _looks_like_natural_text(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) <= 6:
        return False
    math_score = sum(c in _MATH_CHARS for c in text)
    return len(words) > 8 and math_score < 3


def _looks_like_numbered_heading(text: str) -> bool:
    stripped = _strip_formula_delimiters(text)
    if not _SECTION_HEADING_RE.match(stripped):
        return False
    if re.search(r"\\[A-Za-z]+|[_^{}=<>∑∫≤≥≠≈]", stripped):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", stripped)
    return bool(words) and len(words) <= 10


def _looks_like_numbered_paragraph_heading(text: str) -> bool:
    stripped = _strip_formula_delimiters(text)
    if _DISCOURSE_NUMBERED_HEADING_RE.match(stripped):
        return False
    match = _NUMBERED_HEADING_RE.match(stripped)
    if not match:
        return False
    label = match.group("label").strip()
    if re.match(r"^(?:Figure|Fig\.?|Table|Tableau)\b", label, re.I):
        return False
    if re.search(r"\\[A-Za-z]+|[_^{}=<>∑∫≤≥≠≈]", label):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", label)
    if not words or len(words) > 14:
        return False
    if stripped.endswith(".") and len(words) > 8:
        return False
    return True


def _split_numbered_heading_body(block: DocumentBlock, text: str) -> list[DocumentBlock] | None:
    parsed = _parse_embedded_numbered_heading_body(text)
    if parsed is None:
        return None
    title, body = parsed
    if not _looks_like_numbered_paragraph_heading(title):
        return None
    if len(body.split()) < 5:
        return None

    heading = DocumentBlock(
        type="heading",
        text=title,
        page=block.page,
        bbox=block.bbox,
        level=_heading_level_from_numbering(title),
        confidence=min(block.confidence, 0.86),
        metadata={
            **block.metadata,
            "detected_as": "heading",
            "promoted_from": block.type,
            "split_embedded_heading_body": True,
        },
    )
    paragraph = DocumentBlock(
        type="paragraph",
        text=body,
        page=block.page,
        bbox=block.bbox,
        confidence=min(block.confidence, 0.86),
        metadata={
            **block.metadata,
            "split_embedded_heading_body": True,
        },
    )
    return [heading, paragraph]


def _parse_embedded_numbered_heading_body(text: str) -> tuple[str, str] | None:
    stripped = _strip_formula_delimiters(text)
    match = _EMBEDDED_NUMBERED_HEADING_RE.match(stripped)
    if match is not None:
        title = _format_numbered_heading(match.group("number"), match.group("label"))
        body = match.group("body").strip()
        if (
            not _bad_heading_tail_word(title)
            and body[:1].isupper()
            and (len(body.split()) >= 5 or _looks_like_embedded_heading_body(body))
        ):
            return title, body

    numbered = re.match(
        r"^\s*(?P<number>\d+(?:\.\d+)+|[A-Z](?:\.\d+)+|[A-Z]\.)(?:\.)?(?:\s+|(?=[A-Za-zÀ-ÿ]))"
        r"(?P<label>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’(),:;/\-\s]+)$",
        stripped,
        re.I,
    )
    if numbered is None:
        return None

    words = numbered.group("label").split()
    if len(words) < 8:
        return None

    max_title_words = min(9, len(words) - 5)
    for split_at in range(2, max_title_words + 1):
        label = " ".join(words[:split_at]).strip()
        body = " ".join(words[split_at:]).strip()
        if not label or not body:
            continue
        title = _format_numbered_heading(numbered.group("number"), label)
        if not _looks_like_numbered_paragraph_heading(title):
            continue
        if _bad_heading_tail_word(label):
            continue
        if not _looks_like_embedded_heading_body(body):
            continue
        return title, body
    return None


def _bad_heading_tail_word(label: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ]+", label.casefold())
    if not words:
        return True
    return words[-1] in {
        "a",
        "an",
        "the",
        "as",
        "of",
        "for",
        "to",
        "with",
        "in",
        "on",
        "and",
        "or",
        "de",
        "du",
        "des",
        "la",
        "le",
        "les",
        "un",
        "une",
        "et",
        "ou",
    }


def _looks_like_embedded_heading_body(body: str) -> bool:
    clean = re.sub(r"\s+", " ", body or "").strip()
    if len(clean.split()) < 5:
        return False
    if clean[:1].islower():
        return False
    # Labeled paragraph starter: "Encoder: The encoder...", "Decoder: The decoder..."
    if re.match(r"[A-Z][a-zA-ZÀ-ÿ]+:\s", clean):
        return True
    first_words = " ".join(clean.split()[:10]).casefold()
    if re.search(
        r"\b(can|may|must|should|will|is|are|was|were|means|depends|represents?|describes?|"
        r"shows?|uses?|requires?|allows?|gives?|has|have|does|do|peut|peuvent|est|sont|"
        r"signifie|depend|dépend|represente|représente)\b",
        first_words,
    ):
        return True
    return clean.endswith(":")


def _normalize_numbered_heading_text(text: str) -> str | None:
    match = _NUMBERED_HEADING_RE.match(_strip_formula_delimiters(text))
    if not match:
        return None
    return _format_numbered_heading(match.group("number"), match.group("label"))


def _format_numbered_heading(number: str, label: str) -> str:
    clean_number = str(number or "").strip().rstrip(".")
    clean_label = str(label or "").strip()
    if not clean_number:
        return clean_label
    if re.match(r"^[A-Z]$", clean_number, re.I):
        return f"{clean_number}. {clean_label}".strip()
    return f"{clean_number}. {clean_label}".strip()


def _heading_level_from_numbering(text: str) -> int:
    prefix = re.match(r"^\d+(?:\.\d+)*", _strip_formula_delimiters(text))
    if not prefix:
        return 2
    return min(3, max(1, prefix.group(0).count(".") + 1))


def _looks_like_plain_text_label(text: str) -> bool:
    stripped = _strip_formula_delimiters(text)
    if not stripped:
        return False
    if _BIBLIO_MARKER_RE.fullmatch(stripped):
        return True
    if re.search(r"@|https?://|www\.|doi\.org", stripped, re.I):
        return True
    if "\\" in stripped:
        return False
    if re.search(r"[_^=<>∑∫≤≥≠≈√]", stripped):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", stripped)
    if not words:
        return False
    if len(stripped) <= 80 and _SHORT_TEXT_LABEL_RE.fullmatch(stripped):
        return True
    return False


def _semantic_heading_type(block: DocumentBlock) -> str | None:
    if block.type != "heading":
        return None
    text = re.sub(r"\s+", " ", (block.text or "").strip()).strip(" .:")
    if not text:
        return None
    match = re.match(r"^(?P<label>[A-Za-zÀ-ÿ.]+)(?:\s+\d+(?:\.\d+)*)?$", text, re.I)
    if not match:
        return None
    label = match.group("label").casefold()
    # "attention"/"warning" as standalone headings are section titles in English papers, not callouts
    if label in {"attention", "warning"}:
        return None
    return SEMANTIC_PREFIXES.get(label)


def _can_merge_callout_body(heading: DocumentBlock, body: DocumentBlock) -> bool:
    if body.type != "paragraph" or not body.text.strip():
        return False
    if heading.page is not None and body.page is not None and heading.page != body.page:
        return False
    if heading.bbox is None or body.bbox is None:
        return True
    vertical_gap = body.bbox.y0 - heading.bbox.y1
    return -3.0 <= vertical_gap <= max(36.0, heading.bbox.height * 3.0)


def _strip_formula_delimiters(text: str) -> str:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped[1:-1].strip()
    return stripped
