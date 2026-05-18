from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from document.models import DocumentBlock
from document.postprocess.figure_extractor import document_asset_dir
from document.postprocess.latex_quality import latex_looks_corrupt

logger = logging.getLogger("Document.context_assets")

_MATH_TOKEN_RE = re.compile(
    r"(?:\\[A-Za-z]+|[_^]\{?[^{}\s]+\}?|[=<>]|[∑∫√∞≤≥≠∼≈→←⇒⇔±α-ωΑ-Ω]|"
    r"(?<![A-Za-z])\d+\s*/\s*\d+(?![A-Za-z])|(?<![A-Za-z])\d+\s*/\s*[A-Za-z](?![A-Za-z])|"
    r"(?<=[A-Za-z0-9⁰¹²³⁴⁵⁶⁷⁸⁹)\]}])\s*[⋅·×]\s*(?=[A-Za-z0-9({\[]))"
)
_PROSE_WORD_RE = re.compile(r"\b[A-Za-zÀ-ÿ]{3,}\b")
_MATH_WORD_RE = re.compile(r"\b(?:ln|log|exp|sin|cos|tan|lim|sqrt|o)\b", re.I)
_BROKEN_INLINE_LATEX_RE = re.compile(
    r"(?:\\t\s*e\s*x\s*t|\\ma\s*trhm|\\textbf\$\s*\{|\\mathrm\$\s*\{|"
    r"\$[A-Za-z]_\{[^$}]{0,80}\$|\\\s+\\)",
    re.I,
)
_TEXTUAL_TYPES = {
    "paragraph",
    "definition",
    "theorem",
    "example",
    "remark",
    "warning",
    "exercise",
    "question",
}
_REASON_BUDGETS = {
    "inline_math": 12,
    "math_dense_text": 8,
    "low_confidence_text": 12,
}


_CONTEXT_ASSET_RENDER_MODES = {"context_crop_only", "text_with_context_crop"}
_CONTEXT_ASSET_META_KEYS = {
    "context_asset_path",
    "context_asset_type",
    "context_asset_reason",
    "context_asset_display",
    "context_asset_skipped",
    "llm_assets",
}


def _clear_stale_context_asset_metadata(blocks: list[DocumentBlock]) -> None:
    """Strip cached context-asset metadata so each run computes fresh decisions."""
    for block in blocks:
        m = block.metadata
        if not m:
            continue
        if m.get("render_mode") in _CONTEXT_ASSET_RENDER_MODES:
            m.pop("render_mode", None)
        for key in _CONTEXT_ASSET_META_KEYS:
            m.pop(key, None)


def crop_complex_context_blocks(
    pdf_path: str,
    blocks: list[DocumentBlock],
    output_dir: str | Path | None = None,
    max_assets: int = 48,
) -> list[DocumentBlock]:
    """Create visual crops for text blocks whose OCR may be insufficient.

    The crop is kept in metadata so the reader contract stays unchanged. The
    LLM layer can attach the image to an Ollama multimodal request when useful.
    """
    _clear_stale_context_asset_metadata(blocks)

    try:
        import fitz  # type: ignore
    except Exception as exc:
        logger.debug("PyMuPDF indisponible pour assets de contexte: %s", exc)
        return blocks

    path = Path(pdf_path)
    out = Path(output_dir) if output_dir is not None else document_asset_dir(path) / "context"
    out.mkdir(parents=True, exist_ok=True)

    created = 0
    reason_counts: dict[str, int] = {}
    try:
        with fitz.open(path) as doc:
            for index, block in enumerate(blocks):
                if created >= max_assets:
                    break
                if not _needs_context_asset(block):
                    continue
                if not block.bbox or not block.page:
                    continue
                page_index = int(block.page) - 1
                if page_index < 0 or page_index >= len(doc):
                    continue

                page = doc[page_index]
                rect = fitz.Rect(*block.bbox.to_list())
                rect = fitz.Rect(rect.x0 - 12.0, rect.y0 - 10.0, rect.x1 + 12.0, rect.y1 + 10.0)
                rect = fitz.Rect(
                    max(rect.x0, page.rect.x0),
                    max(rect.y0, page.rect.y0),
                    min(rect.x1, page.rect.x1),
                    min(rect.y1, page.rect.y1),
                )
                if rect.is_empty or rect.width <= 8 or rect.height <= 8:
                    continue

                reason = _context_asset_reason(block)
                if _is_unsafe_inline_math_crop_geometry(block, reason, page_width=float(page.rect.width)):
                    block.metadata["context_asset_skipped"] = "wide_inline_math_crop"
                    continue
                if _reason_budget_exceeded(reason, reason_counts):
                    block.metadata["context_asset_skipped"] = f"{reason}_budget"
                    continue

                digest = hashlib.md5(f"{path}-{block.page}-{block.bbox.to_list()}".encode()).hexdigest()[:12]
                image_path = out / f"context_p{block.page}_{index}_{digest}.png"
                if not image_path.exists():
                    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3, 3), alpha=False)
                    pix.save(str(image_path))

                block.metadata["context_asset_path"] = str(image_path)
                block.metadata["context_asset_type"] = "pdf_crop"
                block.metadata["context_asset_reason"] = reason
                if _should_display_context_asset(block, reason):
                    block.metadata["context_asset_display"] = True
                    if _should_replace_text_with_context_asset(block, reason):
                        block.metadata["render_mode"] = "context_crop_only"
                    else:
                        block.metadata.setdefault("render_mode", "text_with_context_crop")
                else:
                    block.metadata["context_asset_display"] = False
                block.metadata["llm_assets"] = [{
                    "type": "image",
                    "path": str(image_path),
                    "reason": reason,
                }]
                created += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    except Exception as exc:
        logger.warning("Création des assets de contexte échouée: %s", exc)

    return blocks


def _reason_budget_exceeded(reason: str, counts: dict[str, int]) -> bool:
    budget = _REASON_BUDGETS.get(reason)
    if budget is None:
        return False
    return counts.get(reason, 0) >= budget


def _needs_context_asset(block: DocumentBlock) -> bool:
    if block.type not in _TEXTUAL_TYPES:
        return False
    if not block.bbox or not block.page:
        return False
    if (block.metadata or {}).get("is_metadata"):
        return False
    if block.metadata.get("context_asset_path"):
        return False

    text = (block.text or block.latex or "").strip()
    if not text:
        return False
    if _looks_like_external_metadata_text(text):
        return False
    if _looks_like_tabular_numeric_data(text):
        return False

    if _looks_like_fragmented_math_text(block):
        return True
    if block.metadata.get("contains_inline_math") or block.metadata.get("formula_mode") in {"inline", "ambiguous"}:
        return True
    if float(block.confidence or 1.0) < 0.74:
        return True

    math_tokens = len(_MATH_TOKEN_RE.findall(text))
    prose_words = len(_PROSE_WORD_RE.findall(text))
    if math_tokens >= 4 and prose_words >= 2:
        return True
    if math_tokens >= 6 and len(text) <= 220:
        return True
    if "$" in text and math_tokens >= 2:
        return True
    return False


def _context_asset_reason(block: DocumentBlock) -> str:
    metadata = block.metadata or {}
    if metadata.get("visual_math_fragment_group") or metadata.get("render_mode") == "context_crop_only":
        return "fragmented_math_text"
    if metadata.get("formula_mode") == "ambiguous" or _looks_like_fragmented_math_text(block):
        return "fragmented_math_text"
    if metadata.get("contains_inline_math") or metadata.get("formula_mode") == "inline":
        return "inline_math"
    if float(block.confidence or 1.0) < 0.74:
        return "low_confidence_text"
    return "math_dense_text"


def _should_display_context_asset(block: DocumentBlock, reason: str) -> bool:
    """Only display a visual crop when the text is genuinely unreadable.

    inline_math, math_dense_text and low_confidence_text all render acceptably
    as text (the app supports LaTeX and the text is usually correct enough).
    Showing a duplicate image crop alongside them causes visual doubling.
    Only fragmented/ambiguous math warrants replacing text with a crop.
    """
    metadata = block.metadata or {}
    if metadata.get("context_asset_display") is False:
        return False
    if _block_is_too_large_for_crop(block):
        return False
    # Only display when the text is too broken to render properly
    if reason == "fragmented_math_text":
        return True
    if metadata.get("formula_mode") == "ambiguous":
        return True
    return False


def _should_replace_text_with_context_asset(block: DocumentBlock, reason: str) -> bool:
    if _block_is_too_large_for_crop(block):
        return False
    return reason == "fragmented_math_text" or (block.metadata or {}).get("formula_mode") == "ambiguous"


def _block_is_too_large_for_crop(block: DocumentBlock) -> bool:
    """Return True if the block covers too large an area to be usefully shown as a crop image.

    A block spanning many lines is normal prose — displaying it as an image
    adds no value and creates visual confusion alongside the text rendering.
    Threshold: bbox height > 120pt (≈ 4–5 lines) or text > 400 chars.
    """
    text = (block.text or block.latex or "").strip()
    if len(text) > 400:
        return True
    if block.bbox is not None and block.bbox.height > 120.0:
        return True
    return False


def _block_has_complex_inline_math(block: DocumentBlock) -> bool:
    """Return True only when the inline math is rich enough that a visual crop adds value."""
    metadata = block.metadata or {}
    text = (block.text or block.latex or "").strip()
    if not text:
        return False
    if metadata.get("formula_mode") in {"ambiguous", "display"}:
        return True
    math_tokens = len(_MATH_TOKEN_RE.findall(text))
    if math_tokens >= 4:
        return True
    if "$" in text and math_tokens >= 2:
        return True
    if re.search(r"\\(?:frac|sum|int|prod|lim|sqrt|begin|matrix)", text):
        return True
    return False


def _is_unsafe_inline_math_crop_geometry(
    block: DocumentBlock,
    reason: str,
    page_width: float | None = None,
) -> bool:
    """Avoid feeding whole-column/page crops to inline math rendering."""
    if reason not in {"inline_math", "math_dense_text"}:
        return False
    if block.bbox is None:
        return False

    metadata = block.metadata or {}
    if metadata.get("mixed_columns_risk"):
        return True

    try:
        width = float(page_width or metadata.get("page_width") or 0.0)
    except (TypeError, ValueError):
        width = 0.0
    if width > 0.0:
        return block.bbox.width >= width * 0.62

    return block.bbox.width >= 430.0


def _looks_like_tabular_numeric_data(text: str) -> bool:
    """Return True when the text is pure numeric row data (e.g. table cells)."""
    if not text:
        return False
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < 4:
        return False
    if len(re.findall(r"[A-Za-zÀ-ÿ]", text)) > 2:
        return False
    return "$" not in text and "\\" not in text


def _looks_like_fragmented_math_text(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    if metadata.get("is_metadata"):
        return False
    if metadata.get("formula_mode") == "ambiguous":
        return True

    text = (block.text or block.latex or "").strip()
    if not text:
        return False
    if _looks_like_external_metadata_text(text):
        return False
    if _looks_like_tabular_numeric_data(text):
        return False

    if text.count("$") % 2 == 1:
        return True
    if re.search(r"\$[A-Za-zÀ-ÿ]{1,2}\$|\bl\s*\$\s*n\b", text):
        return True
    if _BROKEN_INLINE_LATEX_RE.search(text) or latex_looks_corrupt(text):
        return True

    paren_delta = (
        text.count("(") - text.count(")")
        + text.count("[") - text.count("]")
        + text.count("{") - text.count("}")
    )
    math_tokens = len(_MATH_TOKEN_RE.findall(text))
    math_tokens += len(_MATH_WORD_RE.findall(text))
    # Only count bare digits as math signal when real math markers are present;
    # without them, digit-heavy text (e.g. bibliography pages/years) is not math.
    if "$" in text or "\\" in text or math_tokens >= 2:
        math_tokens += len(re.findall(r"\d", text))
    prose_words = len(_PROSE_WORD_RE.findall(text))

    if (
        prose_words >= 18
        and len(text) >= 160
        and "$" not in text
        and "\\" not in text
        and not metadata.get("formula_mode")
    ):
        return False

    if abs(paren_delta) >= 2 and (math_tokens >= 2 or len(text) <= 90):
        return True
    if re.search(r"(?:\(\s*\d|\d\s*\(){2,}", text):
        return True
    if math_tokens >= 7 and prose_words <= 4:
        return True
    if math_tokens >= 5 and re.search(r"\b(?:n|o)\s+(?:n|o)\b", text):
        return True
    return False


def _looks_like_external_metadata_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip().casefold()
    compact = re.sub(r"\s+", "", normalized)
    if not normalized:
        return False
    if re.search(r"\b(?:arxiv|doi|issn|isbn)\s*:", normalized):
        return True
    if "doi.org" in compact or "creativecommons.org" in compact:
        return True
    if re.search(r"https?\s*:\s*/\s*/|www\s*\.", normalized):
        return True
    if normalized.count("/") >= 3 and re.search(r"\b(?:licenses|diagnostics|creativecommons|doi)\b", normalized):
        return True
    return False
