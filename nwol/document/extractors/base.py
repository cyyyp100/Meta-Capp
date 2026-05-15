from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from document.models import DocumentBlock, ExtractionResult


class OptionalBackendUnavailable(RuntimeError):
    """Raised when an optional extraction backend is not installed or configured."""


class BaseExtractor(ABC):
    engine_name = "base"

    @abstractmethod
    def extract(self, pdf_path: str) -> ExtractionResult:
        raise NotImplementedError


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+•]|\d+[\).]|[a-zA-Z][\).]|[IVXLC]+[\).])\s+(.+)$")
_CAPTION_RE = re.compile(r"^(Figure|Fig\.?|Schema|Schéma|Diagramme|Graphique|Illustration|Tableau|Table)\b", re.I)
_DISPLAY_FORMULA_RE = re.compile(r"^\$\$(.+?)\$\$$", re.DOTALL)
_INLINE_FORMULA_RE = re.compile(r"^\$(.+?)\$$", re.DOTALL)


def markdown_to_document_blocks(md_text: str, base_path: str | Path | None = None) -> list[DocumentBlock]:
    """Small, dependency-free Markdown adapter for optional engines."""
    base = Path(base_path) if base_path else None
    blocks: list[DocumentBlock] = []
    paragraph: list[str] = []
    bullets: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(re.sub(r"\s+", " ", line.strip()) for line in paragraph if line.strip()).strip()
        paragraph = []
        if not text:
            return
        display_match = _DISPLAY_FORMULA_RE.match(text)
        inline_match = _INLINE_FORMULA_RE.match(text)
        if display_match or inline_match:
            latex = (display_match or inline_match).group(1).strip()
            blocks.append(DocumentBlock(type="formula", text=latex, latex=latex))
            return
        metadata = {"is_caption": True} if _CAPTION_RE.match(text) else {}
        blocks.append(DocumentBlock(type="paragraph", text=text, metadata=metadata))

    def flush_bullets() -> None:
        nonlocal bullets
        if not bullets:
            return
        text = "\n".join(f"• {item}" for item in bullets)
        blocks.append(DocumentBlock(type="bullet_list", text=text, items=bullets[:]))
        bullets = []

    lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_paragraph()
            i += 1
            continue

        if line.startswith("$$"):
            flush_paragraph()
            flush_bullets()
            formula_lines = [line]
            i += 1
            while i < len(lines) and not "\n".join(formula_lines).strip().endswith("$$"):
                formula_lines.append(lines[i].strip())
                i += 1
            text = "\n".join(formula_lines).strip().strip("$").strip()
            blocks.append(DocumentBlock(type="formula", text=text, latex=text))
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            flush_bullets()
            blocks.append(
                DocumentBlock(
                    type="heading",
                    level=min(3, len(heading_match.group(1))),
                    text=heading_match.group(2).strip(),
                )
            )
            i += 1
            continue

        image_match = _IMAGE_RE.match(line)
        if image_match:
            flush_paragraph()
            flush_bullets()
            image_path = _resolve_path(image_match.group("path"), base)
            blocks.append(
                DocumentBlock(
                    type="figure",
                    image_path=image_path,
                    caption=image_match.group("alt").strip(),
                    text=image_match.group("alt").strip(),
                )
            )
            i += 1
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            flush_paragraph()
            bullets.append(list_match.group(1).strip())
            i += 1
            continue

        flush_bullets()
        paragraph.append(raw)
        i += 1

    flush_paragraph()
    flush_bullets()
    return blocks


def result_from_markdown(
    md_text: str,
    engine_name: str,
    pages: int = 1,
    base_path: str | Path | None = None,
    warnings: list[str] | None = None,
) -> ExtractionResult:
    blocks = markdown_to_document_blocks(md_text, base_path=base_path)
    try:
        from document.postprocess.math_normalizer import normalize_math_blocks

        blocks = normalize_math_blocks(blocks)
    except Exception:
        # Markdown engines already return usable LaTeX most of the time; do not
        # discard a full extraction if a post-processing repair fails.
        pass
    return ExtractionResult(
        blocks=blocks,
        pages=pages,
        score=0.0,  # caller must call update_result_quality() to set the real score
        warnings=list(warnings or []),
        engine_name=engine_name,
        debug_paths=[],
    )


def _resolve_path(raw_path: str, base: Path | None) -> str:
    path = raw_path.strip().split()[0].strip("<>")
    if re.match(r"^[a-z]+://", path, re.I):
        return path
    candidate = Path(path)
    if candidate.is_absolute() or base is None:
        return str(candidate)
    return str((base / candidate).resolve())


def coerce_blocks(blocks: list[dict[str, Any]]) -> list[DocumentBlock]:
    return [DocumentBlock.from_reader_dict(block) for block in blocks]
