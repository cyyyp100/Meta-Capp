# core/scopes.py — Portées texte (Text Scopes)
from __future__ import annotations
import uuid
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("Scopes")

TEXTUAL_BLOCK_TYPES = {
    "paragraph",
    "text",
    "quote",
    "abstract",
    "heading",
    "subheading",
    "subsubheading",
    "definition",
    "theorem",
    "example",
    "remark",
    "warning",
    "code",
}


@dataclass
class TextScope:
    scope_type: str          # "document" | "chapter" | "page" | "block"
    label: str
    page_start: int
    page_end: int
    blocks: list = field(default_factory=list)
    scope_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "scope_id": self.scope_id,
            "type": self.scope_type,
            "label": self.label,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "blocks": self.blocks,
        }

    def plain_text(self) -> str:
        """Retourne le texte brut de tous les blocs (pour le LLM)."""
        parts = []
        for b in self.blocks:
            t = b.get("type", "")
            if t in TEXTUAL_BLOCK_TYPES:
                parts.append(b.get("text", ""))
            elif t == "formula":
                parts.append(b.get("latex") or b.get("text", ""))
            elif t == "bullet_list":
                parts.append("\n".join(str(item) for item in b.get("items", [])))
        return "\n".join(parts)


def make_chapter_scope(title: str, page_start: int, page_end: int,
                       blocks: list) -> TextScope:
    logger.info(f"ChapterScope créé : {title!r} (p. {page_start}–{page_end})")
    return TextScope("chapter", title, page_start, page_end, blocks)


def make_page_scope(page_number: int, blocks: list) -> TextScope:
    return TextScope("page", f"Page {page_number}", page_number, page_number, blocks)
