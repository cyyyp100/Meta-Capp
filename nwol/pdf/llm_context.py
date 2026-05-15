from __future__ import annotations

from typing import Any

from .model import DocumentModel

try:
    from llm.context_builder import build_llm_context as _build_llm_context
except ModuleNotFoundError:
    from nwol.llm.context_builder import build_llm_context as _build_llm_context


def build_llm_context(
    document: DocumentModel,
    current_block_id: str | None = None,
    window: int = 8,
    document_title: str = "",
    current_section: str = "",
) -> dict[str, Any]:
    return _build_llm_context(
        document.to_reader_blocks(),
        current_block_id=current_block_id,
        window=window,
        document_title=document_title,
        current_section=current_section,
    )
