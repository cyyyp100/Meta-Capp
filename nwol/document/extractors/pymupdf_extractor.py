from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from document.extractors.base import BaseExtractor
from document.layout.block_classifier import classify_blocks
from document.layout.column_detector import detect_columns
from document.layout.header_footer import detect_repeated_headers_footers, remove_repeated_headers_footers
from document.layout.math_zone_detector import raw_lines_to_math_aware_blocks
from document.layout.reading_order import order_blocks_for_reading
from document.models import BoundingBox, DocumentBlock, ExtractionResult, RawBlock, RawLine
from document.postprocess.pipeline import postprocess_document_blocks
from document.postprocess.quality import update_result_quality

logger = logging.getLogger("Document.PyMuPDFExtractor")

_REPEATED_HEADER_FOOTER_CACHE: dict[tuple[str, int, int], set[str]] = {}


class PyMuPDFExtractor(BaseExtractor):
    engine_name = "pymupdf_structured"

    def extract(self, pdf_path: str) -> ExtractionResult:
        raw_lines, page_sizes = self.extract_raw_lines(pdf_path)
        warnings: list[str] = []
        if not raw_lines:
            warnings.append("Aucune ligne texte extraite par PyMuPDF.")

        lines, removed = remove_repeated_headers_footers(raw_lines, page_sizes)
        if removed:
            warnings.append(f"{len(removed)} en-tête(s)/pied(s) de page répétitif(s) supprimé(s).")

        raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes=page_sizes)
        layout_warnings = self._layout_warnings(raw_blocks, page_sizes)
        warnings.extend(layout_warnings)

        ordered_raw_blocks = order_blocks_for_reading(raw_blocks, page_sizes)
        blocks = classify_blocks(ordered_raw_blocks)
        blocks = postprocess_document_blocks(blocks, pdf_path, page_sizes)

        for index, block in enumerate(blocks):
            block.metadata.setdefault("engine", self.engine_name)
            block.metadata.setdefault("block_index", index)
            if block.page in page_sizes:
                width, height = page_sizes[block.page]
                block.metadata.setdefault("page_width", width)
                block.metadata.setdefault("page_height", height)

        result = ExtractionResult(
            blocks=blocks,
            pages=len(page_sizes),
            score=0.0,
            warnings=warnings,
            engine_name=self.engine_name,
            debug_paths=[],
        )
        return update_result_quality(result)

    def extract_page(
        self,
        pdf_path: str,
        page_number: int,
        *,
        enrich_assets: bool = True,
    ) -> ExtractionResult:
        if page_number < 1:
            raise ValueError("page_number doit être 1-based et >= 1")

        raw_lines, page_sizes = self.extract_page_raw_lines(pdf_path, page_number)
        warnings: list[str] = []
        if not raw_lines:
            warnings.append(f"Aucune ligne texte extraite par PyMuPDF sur la page {page_number}.")

        repeated = self._document_repeated_header_footer_keys(pdf_path)
        raw_lines, removed = remove_repeated_headers_footers(raw_lines, page_sizes, repeated=repeated)
        if removed:
            warnings.append(f"{len(removed)} en-tête(s)/pied(s) de page répétitif(s) supprimé(s).")

        raw_blocks = raw_lines_to_math_aware_blocks(raw_lines, page_sizes=page_sizes)
        warnings.extend(self._layout_warnings(raw_blocks, page_sizes))

        ordered_raw_blocks = order_blocks_for_reading(raw_blocks, page_sizes)
        blocks = classify_blocks(ordered_raw_blocks)
        blocks = postprocess_document_blocks(
            blocks,
            pdf_path,
            page_sizes,
            enrich_assets=enrich_assets,
            pages={page_number},
        )

        for index, block in enumerate(blocks):
            block.metadata.setdefault("engine", self.engine_name)
            block.metadata.setdefault("block_index", index)
            if block.page in page_sizes:
                width, height = page_sizes[block.page]
                block.metadata.setdefault("page_width", width)
                block.metadata.setdefault("page_height", height)

        result = ExtractionResult(
            blocks=blocks,
            pages=max(page_sizes.keys(), default=page_number),
            score=0.0,
            warnings=warnings,
            engine_name=self.engine_name,
            debug_paths=[],
            metadata={"page_sizes": page_sizes},
        )
        return update_result_quality(result)

    def extract_raw(self, pdf_path: str) -> tuple[list[RawLine], list[RawBlock], dict[int, tuple[float, float]]]:
        raw_lines, page_sizes = self.extract_raw_lines(pdf_path)
        raw_blocks = raw_lines_to_math_aware_blocks(raw_lines, page_sizes=page_sizes)
        return raw_lines, raw_blocks, page_sizes

    def extract_raw_lines(self, pdf_path: str) -> tuple[list[RawLine], dict[int, tuple[float, float]]]:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyMuPDF n'est pas installé. Lance: pip install pymupdf") from exc

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF introuvable: {path}")

        lines: list[RawLine] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                page_sizes[page_index] = (float(page.rect.width), float(page.rect.height))
                page_lines = self._extract_page_lines(page, page_index)
                if not page_lines:
                    page_lines = self._fallback_text_lines(page, page_index)
                lines.extend(page_lines)

        return lines, page_sizes

    def extract_page_raw_lines(self, pdf_path: str, page_number: int) -> tuple[list[RawLine], dict[int, tuple[float, float]]]:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyMuPDF n'est pas installé. Lance: pip install pymupdf") from exc

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF introuvable: {path}")

        with fitz.open(path) as doc:
            if page_number > len(doc):
                return [], {}
            page = doc[page_number - 1]
            page_sizes = {page_number: (float(page.rect.width), float(page.rect.height))}
            page_lines = self._extract_page_lines(page, page_number)
            if not page_lines:
                page_lines = self._fallback_text_lines(page, page_number)
            return page_lines, page_sizes

    def _document_repeated_header_footer_keys(self, pdf_path: str) -> set[str]:
        path = Path(pdf_path)
        try:
            stat = path.stat()
        except OSError:
            return set()

        cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        cached = _REPEATED_HEADER_FOOTER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import fitz  # type: ignore
        except Exception:
            return set()

        margin_lines: list[RawLine] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        try:
            with fitz.open(path) as doc:
                for page_index, page in enumerate(doc, start=1):
                    width = float(page.rect.width)
                    height = float(page.rect.height)
                    page_sizes[page_index] = (width, height)
                    margin = height * 0.12
                    for line in self._extract_page_lines(page, page_index):
                        if line.bbox.y0 <= margin or line.bbox.y1 >= height - margin:
                            margin_lines.append(line)
        except Exception as exc:
            logger.debug("Détection en-têtes/pieds répétitifs ignorée: %s", exc)
            return set()

        repeated = detect_repeated_headers_footers(margin_lines, page_sizes)
        _REPEATED_HEADER_FOOTER_CACHE[cache_key] = repeated

        # Keep the tiny process cache bounded while preserving recent PDFs.
        if len(_REPEATED_HEADER_FOOTER_CACHE) > 12:
            for old_key in list(_REPEATED_HEADER_FOOTER_CACHE)[:-12]:
                _REPEATED_HEADER_FOOTER_CACHE.pop(old_key, None)
        return repeated

    def _extract_page_lines(self, page: Any, page_number: int) -> list[RawLine]:
        data = page.get_text("dict")
        lines: list[RawLine] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for raw_line in block.get("lines", []):
                spans = []
                for span in raw_line.get("spans", []):
                    text = unicodedata.normalize("NFC", str(span.get("text", "")))
                    if not text.strip():
                        continue
                    spans.append({**span, "text": text})
                if not spans:
                    continue
                spans = sorted(spans, key=lambda span: span.get("bbox", [0.0])[0])
                rect = getattr(page, "rect", None)
                page_width = float(getattr(rect, "width", 0.0) or 0.0)
                for segment in self._split_line_spans(spans, page_width):
                    text = self._join_spans(segment).strip()
                    if not text:
                        continue
                    bbox = _span_union_bbox(segment) or BoundingBox.from_seq(raw_line.get("bbox") or block.get("bbox"))
                    font_size = max((float(span.get("size", 0.0)) for span in segment), default=0.0) or None
                    font_name = _dominant_value(str(span.get("font", "")) for span in segment)
                    lines.append(
                        RawLine(
                            text=text,
                            bbox=bbox,
                            page=page_number,
                            font_size=font_size,
                            font_name=font_name,
                            is_bold=any(_span_is_bold(span) for span in segment),
                        )
                    )
        return lines

    def _split_line_spans(self, spans: list[dict[str, Any]], page_width: float) -> list[list[dict[str, Any]]]:
        if len(spans) < 2 or page_width <= 0:
            return [spans]

        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [spans[0]]
        for index in range(1, len(spans)):
            right = spans[index:]
            if _span_gap_suggests_column_split(current, right, page_width):
                segments.append(current)
                current = [spans[index]]
            else:
                current.append(spans[index])
        segments.append(current)
        return segments

    def _fallback_text_lines(self, page: Any, page_number: int) -> list[RawLine]:
        text = page.get_text("text") or ""
        lines: list[RawLine] = []
        y = 0.0
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped:
                y += 12.0
                continue
            lines.append(
                RawLine(
                    text=stripped,
                    bbox=BoundingBox(0.0, y, min(500.0, float(page.rect.width)), y + 10.0),
                    page=page_number,
                    font_size=10.0,
                )
            )
            y += 12.0
        return lines

    def _join_spans(self, spans: list[dict[str, Any]]) -> str:
        if not spans:
            return ""

        sizes = [float(span.get("size", 10.0) or 10.0) for span in spans]
        main_size = max(sizes)

        # Vertical center of "main" spans (normal-size text)
        main_centers = [
            (float(s["bbox"][1]) + float(s["bbox"][3])) / 2
            for s, sz in zip(spans, sizes)
            if sz >= main_size * 0.85 and s.get("bbox")
        ]
        main_center = sum(main_centers) / len(main_centers) if main_centers else 0.0

        text_parts: list[str] = []
        previous_bbox: list[float] | None = None
        previous_math_font = False
        previous_text = ""
        for span, size in zip(spans, sizes):
            text = unicodedata.normalize("NFC", str(span.get("text", "")))
            bbox = span.get("bbox")
            current_math_font = _span_is_math_font(span)

            is_script = bool(bbox) and size < main_size * 0.75
            span_center = (float(bbox[1]) + float(bbox[3])) / 2 if bbox else main_center
            is_super = is_script and span_center < main_center - main_size * 0.15
            is_sub = is_script and not is_super and span_center > main_center + main_size * 0.15

            if previous_bbox and bbox and text_parts:
                gap = float(bbox[0]) - float(previous_bbox[2])
                if gap > max(3.0, size * 0.25) and not text_parts[-1].endswith(" ") and not (is_super or is_sub):
                    if previous_math_font or current_math_font:
                        if _needs_math_prose_space(previous_text, text, previous_math_font, current_math_font):
                            text_parts.append(" ")
                    else:
                        spaces = max(1, min(8, int(gap / max(size * 0.45, 1.0))))
                        text_parts.append(" " * spaces)

            if is_super:
                text_parts.append(f"^{{{text.strip()}}}")
            elif is_sub:
                text_parts.append(f"_{{{text.strip()}}}")
            else:
                text_parts.append(text)

            previous_bbox = bbox
            previous_math_font = current_math_font
            previous_text = text
        return "".join(text_parts)

    def _layout_warnings(self, raw_blocks: list[RawBlock], page_sizes: dict[int, tuple[float, float]]) -> list[str]:
        two_column_pages = 0
        for page, size in page_sizes.items():
            page_blocks = [block for block in raw_blocks if block.page == page]
            layout = detect_columns(page_blocks, page_width=size[0])
            if layout.layout_type == "two_columns":
                two_column_pages += 1
        if two_column_pages:
            return [f"Layout deux colonnes détecté sur {two_column_pages} page(s), ordre de lecture corrigé."]
        return []


def _span_is_bold(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).casefold()
    return "bold" in font or "black" in font or "semibold" in font


def _span_is_math_font(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).casefold()
    return any(marker in font for marker in ("symbol", "math", "cmmi", "cmsy", "cmex", "stix"))


def _needs_math_prose_space(previous_text: str, current_text: str, previous_math_font: bool, current_math_font: bool) -> bool:
    if previous_math_font == current_math_font:
        return False
    previous = previous_text.strip()
    current = current_text.strip()
    if not previous or not current:
        return False
    if previous[-1:] in "([{/" or current[:1] in ")]},.;:/":
        return False
    return _looks_like_prose_span(previous) or _looks_like_prose_span(current)


def _looks_like_prose_span(text: str) -> bool:
    cleaned = re.sub(r"[^A-Za-zÀ-ÿ]", "", text)
    return len(cleaned) >= 2


def _span_gap_suggests_column_split(
    left_spans: list[dict[str, Any]],
    right_spans: list[dict[str, Any]],
    page_width: float,
) -> bool:
    if not left_spans or not right_spans:
        return False
    left_last = _span_bbox(left_spans[-1])
    right_first = _span_bbox(right_spans[0])
    if left_last is None or right_first is None:
        return False

    gap = right_first[0] - left_last[2]
    max_size = max(float(span.get("size", 0.0) or 0.0) for span in [*left_spans, right_spans[0]])
    if gap < max(14.0, max_size * 1.05):
        return False
    if not (left_last[2] <= page_width * 0.53 and right_first[0] >= page_width * 0.47):
        return False

    left_bbox = _span_union_bbox(left_spans)
    right_bbox = _span_union_bbox(right_spans)
    if left_bbox is None or right_bbox is None:
        return False
    if left_bbox.width < page_width * 0.12 or right_bbox.width < page_width * 0.12:
        return False

    left_words = _span_word_count(left_spans)
    right_words = _span_word_count(right_spans)
    if left_words < 2 or right_words < 2:
        return False
    return True


def _span_bbox(span: dict[str, Any]) -> list[float] | None:
    raw = span.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    except (TypeError, ValueError):
        return None


def _span_union_bbox(spans: list[dict[str, Any]]) -> BoundingBox | None:
    boxes = [_span_bbox(span) for span in spans]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    bbox = BoundingBox.from_seq(boxes[0])
    for box in boxes[1:]:
        bbox = bbox.union(BoundingBox.from_seq(box))
    return bbox


def _span_word_count(spans: list[dict[str, Any]]) -> int:
    text = " ".join(str(span.get("text", "")) for span in spans)
    return len(re.findall(r"[A-Za-zÀ-ÿ0-9]{2,}", text))


def _dominant_value(values: Any) -> str | None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    return max(set(cleaned), key=cleaned.count)
