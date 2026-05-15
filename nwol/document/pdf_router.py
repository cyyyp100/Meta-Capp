from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from collections import OrderedDict
from pathlib import Path
from typing import Any

from document.extractors.base import OptionalBackendUnavailable
from document.extractors.opendataloader_extractor import OpenDataLoaderExtractor
from document.extractors.pymupdf_extractor import PyMuPDFExtractor
from document.layout.reading_order import order_blocks_for_reading
from document.models import BoundingBox, DocumentBlock, ExtractionResult
from document.postprocess.figure_extractor import blocks_have_missing_managed_assets
from document.postprocess.learning_normalizer import normalize_for_learning
from document.postprocess.learning_chunks import (
    build_learning_chunks,
    detect_document_type,
    enrich_blocks_for_learning,
    is_geometrically_valid,
    text_similarity,
)
from document.postprocess.quality import update_result_quality

logger = logging.getLogger("Document.router")

QUALITY_THRESHOLD = 0.72

_result_cache: OrderedDict[tuple[str, float, str], ExtractionResult] = OrderedDict()
_CACHE_MAX_SIZE = 8


def extract_document(pdf_path: str, preferred_engine: str = "auto") -> ExtractionResult:
    path = str(Path(pdf_path).resolve())

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    engine = (preferred_engine or "auto").strip().casefold().replace("-", "_")
    key = (path, mtime, engine)

    if key in _result_cache:
        _result_cache.move_to_end(key)
        return _result_cache[key]

    result = _run_extraction(path, engine)

    if len(_result_cache) >= _CACHE_MAX_SIZE:
        _result_cache.popitem(last=False)

    _result_cache[key] = result
    return result


def _run_extraction(pdf_path: str, preferred_engine: str = "auto") -> ExtractionResult:
    engine = (preferred_engine or "auto").strip().casefold().replace("-", "_")

    if engine in ("auto", "default", ""):
        return _run_auto(pdf_path)

    if engine in ("opendataloader", "open_data_loader", "opendataloader_pdf", "odl"):
        return _run_opendataloader_then_fallback(pdf_path)

    if engine in ("pymupdf", "pymupdf_structured", "fitz"):
        return _run_pymupdf(pdf_path)

    if engine == "scientific":
        return _run_scientific_pipeline(pdf_path)

    if engine == "marker":
        return _run_marker_then_fallback(pdf_path)

    logger.warning("Moteur PDF inconnu '%s', mode auto utilisé.", preferred_engine)
    return _run_auto(pdf_path)


def _run_auto(pdf_path: str) -> ExtractionResult:
    candidates: list[ExtractionResult] = []

    pymupdf = _run_pymupdf(pdf_path)
    candidates.append(pymupdf)

    opendataloader = _try_opendataloader(pdf_path)
    if opendataloader is not None:
        candidates.append(opendataloader)

    if pymupdf.blocks:
        doc_type = detect_document_type(pymupdf.blocks, pages=pymupdf.pages)
        semantic_results = [result for result in (opendataloader,) if result is not None]

        fused = _fuse_geometric_and_semantic_results(
            pymupdf,
            semantic_results,
            document_type=doc_type,
        )
        candidates.append(fused)

        if _result_quality_acceptable(fused):
            return fused

        logger.warning(
            "Pipeline fusionné jugé insuffisant "
            "(score=%s, blocs=%s, pages_couvertes=%s/%s, type=%s), sélection du meilleur backend.",
            fused.score,
            len(fused.blocks),
            len(_covered_pages(fused)),
            fused.pages,
            doc_type,
        )

    return _best_result(candidates)


def _run_scientific_pipeline(pdf_path: str) -> ExtractionResult:
    pymupdf = _run_pymupdf(pdf_path)
    opendataloader = _try_opendataloader(pdf_path)

    if not pymupdf.blocks:
        return opendataloader or pymupdf

    semantic_results: list[ExtractionResult] = []
    if opendataloader is not None:
        semantic_results.append(opendataloader)

    doc_type = detect_document_type(pymupdf.blocks, pages=pymupdf.pages)
    if doc_type == "course_simple":
        doc_type = "scientific_article"
    return _fuse_geometric_and_semantic_results(
        pymupdf,
        semantic_results,
        document_type=doc_type,
    )


def _run_opendataloader_then_fallback(pdf_path: str) -> ExtractionResult:
    result = _try_opendataloader(pdf_path)

    if result is not None and _result_quality_acceptable(result):
        return result

    if result is not None:
        logger.warning(
            "OpenDataLoader demandé mais jugé insuffisant "
            "(score=%s, blocs=%s, pages_couvertes=%s/%s), repli sur PyMuPDF.",
            result.score,
            len(result.blocks),
            len(_covered_pages(result)),
            result.pages,
        )

    fallback = _run_pymupdf(pdf_path)

    if fallback.blocks:
        return _best_result([candidate for candidate in (result, fallback) if candidate is not None])

    return result or fallback


def _try_opendataloader(pdf_path: str) -> ExtractionResult | None:
    try:
        extractor = OpenDataLoaderExtractor()
        result = extractor.extract(pdf_path)
        _tag_result(result)

        logger.info(
            "[opendataloader_pdf] %s bloc(s), score=%s, pages_couvertes=%s/%s",
            len(result.blocks),
            result.score,
            sorted(_covered_pages(result)),
            result.pages,
        )

        return result

    except OptionalBackendUnavailable as exc:
        logger.info("OpenDataLoader indisponible, repli possible sur PyMuPDF : %s", exc)

    except Exception as exc:
        logger.error("OpenDataLoader échoué, repli possible sur PyMuPDF : %s", exc)

    return None


def _run_marker_then_fallback(pdf_path: str) -> ExtractionResult:
    result = _try_marker(pdf_path)
    if result is not None:
        return result

    return _run_pymupdf(pdf_path)


def _try_marker(pdf_path: str) -> ExtractionResult | None:
    try:
        from document.extractors.marker_extractor import MarkerExtractor

        extractor = MarkerExtractor()
        result = extractor.extract(pdf_path)
        _tag_result(result)

        logger.info("[marker] %s bloc(s), score=%s", len(result.blocks), result.score)

        return result

    except OptionalBackendUnavailable as exc:
        logger.warning("Marker indisponible : %s", exc)

    except Exception as exc:
        logger.error("Marker échoué : %s", exc)

    return None


def _run_pymupdf(pdf_path: str) -> ExtractionResult:
    extractor = PyMuPDFExtractor()

    try:
        result = extractor.extract(pdf_path)
        _tag_result(result)

        logger.info(
            "[pymupdf_structured] %s bloc(s), score=%s, pages_couvertes=%s/%s",
            len(result.blocks),
            result.score,
            sorted(_covered_pages(result)),
            result.pages,
        )

        return result

    except OptionalBackendUnavailable as exc:
        logger.error("PyMuPDF indisponible : %s", exc)

    except Exception as exc:
        logger.error("Extraction PyMuPDF échouée : %s", exc)

    return ExtractionResult(
        blocks=[],
        pages=0,
        score=0.0,
        warnings=["Extraction PDF impossible (PyMuPDF indisponible)."],
        engine_name="none",
        debug_paths=[],
    )


def _tag_result(result: ExtractionResult) -> None:
    for index, block in enumerate(result.blocks):
        block.metadata.setdefault("engine", result.engine_name)
        block.metadata.setdefault("block_index", index)

    result.blocks = normalize_for_learning(result.blocks)
    doc_type = str(result.metadata.get("document_type") or detect_document_type(result.blocks, pages=result.pages))
    result.metadata["document_type"] = doc_type
    result.blocks = enrich_blocks_for_learning(result.blocks, document_type=doc_type)


def _fuse_geometric_and_semantic_results(
    geo_result: ExtractionResult,
    semantic_results: list[ExtractionResult],
    *,
    document_type: str,
) -> ExtractionResult:
    geo_blocks = [_copy_block(block) for block in geo_result.blocks]
    semantic_blocks = [
        block
        for result in semantic_results
        for block in result.blocks
    ]

    matched_semantic_ids: set[int] = set()
    replacements = 0
    for geo_block in geo_blocks:
        geo_block.metadata.setdefault("geometry_source", geo_result.engine_name)
        match = _best_semantic_match(geo_block, semantic_blocks, matched_semantic_ids)
        if match is None:
            continue
        matched_semantic_ids.add(id(match))
        if _maybe_replace_text(geo_block, match):
            replacements += 1

    appended = 0
    for semantic_block in semantic_blocks:
        if id(semantic_block) in matched_semantic_ids:
            continue
        if not _semantic_block_can_be_appended(semantic_block, geo_blocks):
            continue
        clone = _copy_block(semantic_block)
        clone.metadata.setdefault("geometry_source", clone.metadata.get("engine") or clone.metadata.get("source") or "semantic")
        clone.metadata.setdefault("semantic_only_block", True)
        geo_blocks.append(clone)
        appended += 1

    geo_blocks = _order_blocks_for_reading(geo_blocks)
    warnings = _unique([
        *geo_result.warnings,
        *[
            warning
            for result in semantic_results
            for warning in result.warnings
        ],
    ])
    if replacements:
        warnings.append(f"{replacements} bloc(s) enrichi(s) par extraction sémantique.")
    if appended:
        warnings.append(f"{appended} bloc(s) sémantique(s) ajouté(s) avec géométrie exploitable.")

    result = ExtractionResult(
        blocks=geo_blocks,
        pages=max([geo_result.pages, *[result.pages for result in semantic_results]] or [geo_result.pages]),
        score=geo_result.score,
        warnings=warnings,
        engine_name=_fused_engine_name(geo_result, semantic_results),
        debug_paths=_unique([
            *geo_result.debug_paths,
            *[
                path
                for result in semantic_results
                for path in result.debug_paths
            ],
        ]),
        metadata={
            "document_type": document_type,
            "geometry_engine": geo_result.engine_name,
            "semantic_engines": [result.engine_name for result in semantic_results],
            "semantic_replacements": replacements,
            "semantic_appended_blocks": appended,
        },
    )
    update_result_quality(result)
    _tag_result(result)
    return result


def _best_semantic_match(
    geo_block: DocumentBlock,
    semantic_blocks: list[DocumentBlock],
    matched_ids: set[int],
) -> DocumentBlock | None:
    if geo_block.page is None:
        return None
    geo_text = _block_text(geo_block)
    if not geo_text.strip():
        return None

    scored: list[tuple[float, DocumentBlock]] = []
    for semantic_block in semantic_blocks:
        if id(semantic_block) in matched_ids:
            continue
        if semantic_block.page is not None and semantic_block.page != geo_block.page:
            continue
        if not _compatible_block_types(geo_block, semantic_block):
            continue
        semantic_text = _block_text(semantic_block)
        if not semantic_text.strip():
            continue
        similarity = text_similarity(geo_text, semantic_text)
        if similarity <= 0.55:
            continue
        score = similarity
        if semantic_block.bbox is not None and geo_block.bbox is not None:
            score += _bbox_similarity_bonus(geo_block.bbox, semantic_block.bbox)
        if _semantic_text_is_richer(geo_text, semantic_text):
            score += 0.08
        scored.append((score, semantic_block))

    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _maybe_replace_text(geo_block: DocumentBlock, semantic_block: DocumentBlock) -> bool:
    if geo_block.page != semantic_block.page and semantic_block.page is not None:
        return False
    if geo_block.bbox is None:
        return False

    geo_text = _block_text(geo_block)
    semantic_text = _block_text(semantic_block)
    if text_similarity(geo_text, semantic_text) <= 0.55:
        return False
    if _semantic_heading_prepends_section_to_paragraph(geo_block, semantic_block, geo_text, semantic_text):
        _merge_semantic_metadata(geo_block, semantic_block)
        return False
    _maybe_upgrade_to_heading(geo_block, semantic_block, geo_text)
    if _semantic_replacement_prepends_unrelated_text(geo_text, semantic_text):
        _merge_semantic_metadata(geo_block, semantic_block)
        return False
    if _semantic_text_has_corrupt_inline_math(semantic_text) and not _semantic_text_has_corrupt_inline_math(geo_text):
        _merge_semantic_metadata(geo_block, semantic_block)
        return False
    if not _semantic_text_is_richer(geo_text, semantic_text):
        _merge_semantic_metadata(geo_block, semantic_block)
        return False

    if geo_block.text and semantic_text:
        geo_block.metadata.setdefault("original_text", geo_block.text)
        geo_block.text = semantic_text
    if geo_block.type == "formula" and semantic_block.latex:
        geo_block.latex = semantic_block.latex
    if geo_block.type == "table":
        geo_block.markdown = semantic_block.markdown or geo_block.markdown
        geo_block.html = semantic_block.html or geo_block.html
    if semantic_block.caption and not geo_block.caption:
        geo_block.caption = semantic_block.caption
    geo_block.confidence = max(float(geo_block.confidence or 0.0), float(semantic_block.confidence or 0.0))
    _merge_semantic_metadata(geo_block, semantic_block)
    geo_block.metadata["text_enriched_by"] = (
        semantic_block.metadata.get("engine")
        or semantic_block.metadata.get("source")
        or "semantic"
    )
    geo_block.metadata["semantic_block_id"] = semantic_block.id
    return True


def _maybe_upgrade_to_heading(
    geo_block: DocumentBlock,
    semantic_block: DocumentBlock,
    geo_text: str,
) -> None:
    """Upgrade geo block type from paragraph/text to heading when the semantic source confirms it.

    Only fires when: semantic engine says heading, geo engine says paragraph/text, and
    the text is short enough to plausibly be a heading (≤200 chars).  Never downgrades.
    """
    if semantic_block.type != "heading":
        return
    if geo_block.type not in ("paragraph", "text"):
        return
    if len(geo_text.strip()) > 200:
        return
    geo_block.type = "heading"
    if semantic_block.level is not None:
        geo_block.level = semantic_block.level
    semantic_source = (
        semantic_block.metadata.get("engine")
        or semantic_block.metadata.get("source")
        or "semantic"
    )
    geo_block.metadata["type_upgraded_to_heading_by"] = semantic_source


def _merge_semantic_metadata(geo_block: DocumentBlock, semantic_block: DocumentBlock) -> None:
    for key, value in (semantic_block.metadata or {}).items():
        if key in {"block_index", "engine"}:
            continue
        geo_block.metadata.setdefault(key, value)
    semantic_source = semantic_block.metadata.get("engine") or semantic_block.metadata.get("source")
    if semantic_source:
        sources = list(geo_block.metadata.get("semantic_sources") or [])
        if semantic_source not in sources:
            sources.append(semantic_source)
        geo_block.metadata["semantic_sources"] = sources


def _semantic_text_is_richer(geo_text: str, semantic_text: str) -> bool:
    geo_clean = _clean_text_for_comparison(geo_text)
    semantic_clean = _clean_text_for_comparison(semantic_text)
    if len(semantic_clean) > len(geo_clean) * 1.08:
        return True
    if semantic_clean.count("$") >= geo_clean.count("$") + 2:
        return True
    if "�" in geo_clean and "�" not in semantic_clean:
        return True
    return False


def _semantic_text_has_corrupt_inline_math(text: str) -> bool:
    """Detect ODL snippets where math-font words were split with stray dollar signs."""
    cleaned = _clean_text_for_comparison(text)
    if not cleaned:
        return False
    return bool(re.search(r"(?<=[A-Za-zÀ-ÿ])\$(?=[A-Za-zÀ-ÿ])", cleaned))


def _semantic_heading_prepends_section_to_paragraph(
    geo_block: DocumentBlock,
    semantic_block: DocumentBlock,
    geo_text: str,
    semantic_text: str,
) -> bool:
    if semantic_block.type != "heading":
        return False
    if geo_block.type not in {"paragraph", "text", "abstract", "definition", "theorem", "example", "remark", "warning"}:
        return False

    geo_clean = _clean_text_for_comparison(geo_text).casefold()
    semantic_clean = _clean_text_for_comparison(semantic_text).casefold()
    if not geo_clean or not semantic_clean:
        return False
    if _starts_with_section_number(geo_clean) or not _starts_with_section_number(semantic_clean):
        return False

    offset = semantic_clean.find(geo_clean[: min(70, max(24, len(geo_clean) // 2))])
    if offset <= 0:
        return False
    prefix = semantic_clean[:offset].strip(" .:-")
    prefix_without_number = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*|[a-z](?:\.\d+)*\.?)(?:\s+|(?=[a-z]))",
        "",
        prefix,
        flags=re.I,
    ).strip()
    if len(re.findall(r"[a-zÀ-ÿ]{2,}", prefix_without_number, re.I)) < 2:
        return False
    return _looks_like_sentence_body(geo_clean)


def _starts_with_section_number(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*\.?)(?:\s+|(?=[A-Za-zÀ-ÿ]))",
            text,
            re.I,
        )
    )


def _looks_like_sentence_body(text: str) -> bool:
    words = text.split()
    if len(words) < 5:
        return False
    first_words = " ".join(words[:10])
    return bool(
        re.search(
            r"\b(can|may|must|should|will|is|are|was|were|means|depends|represents?|describes?|"
            r"shows?|uses?|requires?|allows?|gives?|has|have|does|do|peut|peuvent|est|sont|"
            r"signifie|depend|dépend|represente|représente)\b",
            first_words,
            re.I,
        )
    )


def _semantic_replacement_prepends_unrelated_text(geo_text: str, semantic_text: str) -> bool:
    geo_clean = _clean_text_for_comparison(geo_text).casefold()
    semantic_clean = _clean_text_for_comparison(semantic_text).casefold()
    if len(geo_clean) < 80 or len(semantic_clean) <= len(geo_clean) * 1.25:
        return False
    prefix = geo_clean[: min(90, max(40, len(geo_clean) // 3))]
    if semantic_clean.startswith(prefix):
        return False
    offset = semantic_clean.find(prefix)
    return offset > 80


def _semantic_block_can_be_appended(
    semantic_block: DocumentBlock,
    existing_blocks: list[DocumentBlock],
) -> bool:
    if semantic_block.metadata.get("is_reference") or semantic_block.metadata.get("is_metadata"):
        return False
    if not is_geometrically_valid(semantic_block):
        return False
    if semantic_block.type in {"figure", "table", "formula"}:
        return not _has_overlapping_block(semantic_block, existing_blocks)
    text = _block_text(semantic_block).strip()
    if len(text) < 40:
        return False
    return not _has_similar_text_block(semantic_block, existing_blocks)


def _has_similar_text_block(block: DocumentBlock, existing_blocks: list[DocumentBlock]) -> bool:
    text = _block_text(block)
    for existing in existing_blocks:
        if block.page is not None and existing.page is not None and block.page != existing.page:
            continue
        if text_similarity(text, _block_text(existing)) > 0.72:
            return True
    return False


def _has_overlapping_block(block: DocumentBlock, existing_blocks: list[DocumentBlock]) -> bool:
    if block.bbox is None:
        return False
    for existing in existing_blocks:
        if existing.bbox is None:
            continue
        if block.page is not None and existing.page is not None and block.page != existing.page:
            continue
        if _bbox_iou(block.bbox, existing.bbox) > 0.45:
            return True
    return False


def _compatible_block_types(left: DocumentBlock, right: DocumentBlock) -> bool:
    if left.type == right.type:
        return True
    # Allow a semantic heading to match a geometric paragraph/text (type-upgrade path).
    if right.type == "heading" and left.type in ("paragraph", "text"):
        return True
    textual = {"paragraph", "text", "abstract", "definition", "theorem", "example", "remark", "warning"}
    return left.type in textual and right.type in textual


def _copy_block(block: DocumentBlock) -> DocumentBlock:
    return DocumentBlock(
        type=block.type,
        text=block.text,
        page=block.page,
        bbox=BoundingBox.from_seq(block.bbox.to_list()) if block.bbox else None,
        level=block.level,
        items=list(block.items) if block.items is not None else None,
        latex=block.latex,
        html=block.html,
        markdown=block.markdown,
        image_path=block.image_path,
        caption=block.caption,
        confidence=block.confidence,
        metadata=dict(block.metadata or {}),
        id=block.id,
    )


def _bbox_similarity_bonus(left: BoundingBox, right: BoundingBox) -> float:
    iou = _bbox_iou(left, right)
    if iou:
        return min(0.2, iou * 0.2)
    center_delta = abs(left.center_x - right.center_x) + abs(left.center_y - right.center_y)
    return max(0.0, 0.1 - center_delta / 2000.0)


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    ix0, iy0 = max(left.x0, right.x0), max(left.y0, right.y0)
    ix1, iy1 = min(left.x1, right.x1), min(left.y1, right.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = left.width * left.height + right.width * right.height - inter
    return inter / union if union > 0 else 0.0


def _clean_text_for_comparison(text: str) -> str:
    return " ".join(str(text or "").split())


def _fused_engine_name(geo_result: ExtractionResult, semantic_results: list[ExtractionResult]) -> str:
    semantic_names = [result.engine_name for result in semantic_results if result.blocks]
    if not semantic_names:
        return geo_result.engine_name
    return "+".join([geo_result.engine_name, *semantic_names])


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if item))


def _position_key(block: DocumentBlock) -> tuple[int, float, float]:
    bbox = block.bbox
    return (
        int(block.page or 0),
        bbox.y0 if bbox else float("inf"),
        bbox.x0 if bbox else float("inf"),
    )


def _order_blocks_for_reading(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    page_sizes = _page_sizes_from_blocks(blocks)
    try:
        return list(order_blocks_for_reading(blocks, page_sizes))  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("Ordre de lecture géométrique indisponible, repli positionnel: %s", exc)
        # Preserve geometric order already stamped if available.
        if any(b.metadata.get("reading_order_index") is not None for b in blocks):
            def _by_reading_order(b: DocumentBlock) -> float:
                idx = b.metadata.get("reading_order_index")
                return float(idx) if idx is not None else float("inf")
            return sorted(blocks, key=_by_reading_order)
        return sorted(blocks, key=_position_key)


def _page_sizes_from_blocks(blocks: list[DocumentBlock]) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    for block in blocks:
        if block.page is None:
            continue
        metadata = block.metadata or {}
        try:
            width = float(metadata.get("page_width") or 0.0)
            height = float(metadata.get("page_height") or 0.0)
        except (TypeError, ValueError):
            continue
        if width > 0.0 and height > 0.0:
            sizes[int(block.page)] = (width, height)
    return sizes

def _block_is_meaningful(block: Any) -> bool:
    if _block_text(block).strip():
        return True

    if getattr(block, "type", None) == "figure":
        return bool(
            getattr(block, "image_path", None)
            or getattr(block, "caption", None)
        )

    if getattr(block, "type", None) == "formula":
        metadata = getattr(block, "metadata", None) or {}
        return bool(
            getattr(block, "latex", None)
            or getattr(block, "image_path", None)
            or metadata.get("formula_image_path")
            or metadata.get("render_mode") == "pdf_crop"
        )

    if getattr(block, "type", None) == "table":
        return bool(
            getattr(block, "markdown", None)
            or getattr(block, "html", None)
        )

    metadata = getattr(block, "metadata", None) or {}
    return bool(
        metadata.get("context_asset_path")
        or metadata.get("llm_assets")
    )
    
def _result_quality_acceptable(result: ExtractionResult) -> bool:
    if not result.blocks:
        logger.debug("Qualité refusée: aucun bloc.")
        return False

    if result.score < QUALITY_THRESHOLD:
        logger.debug("Qualité refusée: score=%s < seuil=%s.", result.score, QUALITY_THRESHOLD)
        return False

    meaningful_blocks = [
        block for block in result.blocks
        if _block_is_meaningful(block)
    ]

    if not meaningful_blocks:
        logger.debug("Qualité refusée: aucun bloc significatif.")
        return False

    meaningful_rate = len(meaningful_blocks) / max(len(result.blocks), 1)

    # Avant tu rejetais si empty_rate > 0.35.
    # C'était trop strict pour OpenDataLoader, car certains blocs image/formule
    # peuvent être utiles même sans texte brut.
    if meaningful_rate < 0.45:
        logger.debug(
            "Qualité refusée: meaningful_rate=%s, meaningful=%s/%s.",
            round(meaningful_rate, 3),
            len(meaningful_blocks),
            len(result.blocks),
        )
        return False

    geometric_blocks = [
        block for block in meaningful_blocks
        if is_geometrically_valid(block)
    ]
    geometric_rate = len(geometric_blocks) / max(len(meaningful_blocks), 1)
    if result.pages >= 2 and geometric_rate < 0.35:
        logger.debug(
            "Qualité refusée: geometric_rate=%s, geometric=%s/%s.",
            round(geometric_rate, 3),
            len(geometric_blocks),
            len(meaningful_blocks),
        )
        return False

    covered_pages = _covered_pages(result)

    if result.pages >= 3 and covered_pages:
        min_covered = max(1, min(result.pages, result.pages // 2))
        if len(covered_pages) < min_covered:
            logger.debug(
                "Qualité refusée: pages couvertes=%s/%s, minimum=%s.",
                len(covered_pages),
                result.pages,
                min_covered,
            )
            return False

    critical_warnings = [
        warning for warning in result.warnings
        if "aucun bloc" in warning.casefold()
    ]

    if critical_warnings:
        logger.debug("Qualité refusée: warnings critiques=%s.", critical_warnings)
        return False

    return True


def _best_result(candidates: list[ExtractionResult]) -> ExtractionResult:
    usable = [result for result in candidates if result.blocks]

    if not usable:
        return candidates[-1]

    return max(
        usable,
        key=lambda result: (
            _result_quality_acceptable(result),
            result.score,
            len(_covered_pages(result)),
            len(result.blocks),
        ),
    )


def _covered_pages(result: ExtractionResult) -> set[int]:
    return {int(block.page or 0) for block in result.blocks if block.page}


def _block_text(block: Any) -> str:
    if getattr(block, "type", None) == "bullet_list":
        return " ".join(getattr(block, "items", None) or [])

    if getattr(block, "type", None) == "formula":
        return getattr(block, "latex", None) or getattr(block, "text", "") or ""

    if getattr(block, "type", None) == "table":
        return (
            getattr(block, "text", None)
            or getattr(block, "markdown", None)
            or getattr(block, "html", None)
            or ""
        )

    return getattr(block, "text", None) or getattr(block, "caption", None) or ""


def clear_cache() -> None:
    _result_cache.clear()


def export_debug_json(result: ExtractionResult, output_path: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path else Path("debug_blocks.json")

    data = {
        "engine": result.engine_name,
        "pages": result.pages,
        "score": result.score,
        "warnings": result.warnings,
        "metadata": result.metadata,
        "pages_covered": sorted(_covered_pages(result)),
        "learning_chunks": [
            {
                "id": chunk.id,
                "type": chunk.chunk_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "quality_score": chunk.quality_score,
                "generation_mode": chunk.generation_mode,
                "source_blocks": chunk.source_blocks,
                "visual_assets": [asset.to_dict() for asset in chunk.visual_assets],
                "text": chunk.text[:240],
            }
            for chunk in build_learning_chunks(result.blocks)
        ],
        "blocks": [
            {
                "id": b.id,
                "type": b.type,
                "page": b.page,
                "confidence": b.confidence,
                "quality_score": b.metadata.get("quality_score"),
                "displayable": b.metadata.get("displayable"),
                "generation_mode": b.metadata.get("generation_mode"),
                "text": (b.text or "")[:200],
                "latex": b.latex,
                "items": b.items,
                "caption": b.caption,
                "image_path": b.image_path,
                "bbox": b.bbox.to_list() if b.bbox else None,
                "metadata": b.metadata,
            }
            for b in result.blocks
        ],
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Debug JSON exporté : %s (%d blocs)", path, len(result.blocks))

    return path


def compare_pdf_backends(pdf_path: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run each backend independently and return compact debug metrics."""
    path = str(Path(pdf_path).resolve())
    report: dict[str, Any] = {}

    engines = ["opendataloader", "pymupdf"]
    if (os.environ.get("NWOL_DEBUG_MARKER") or "").casefold() in {"1", "true", "yes", "on"}:
        engines.append("marker")

    for engine in engines:
        try:
            result = _run_backend_strict(path, engine)
            _tag_result(result)
            report[engine] = _summarize_backend_result(
                path,
                engine,
                result,
                output_dir=output_dir,
            )

        except OptionalBackendUnavailable as exc:
            report[engine] = {
                "available": False,
                "engine_name": engine,
                "score": 0.0,
                "warnings": [str(exc)],
                "blocks": 0,
            }

        except Exception as exc:
            report[engine] = {
                "available": False,
                "engine_name": engine,
                "score": 0.0,
                "warnings": [f"{engine} a échoué: {exc}"],
                "blocks": 0,
            }

    return report


def _run_backend_strict(pdf_path: str, engine: str) -> ExtractionResult:
    if engine == "opendataloader":
        return OpenDataLoaderExtractor().extract(pdf_path)

    if engine == "pymupdf":
        return PyMuPDFExtractor().extract(pdf_path)

    if engine == "marker":
        from document.extractors.marker_extractor import MarkerExtractor

        return MarkerExtractor().extract(pdf_path)

    raise ValueError(f"Moteur inconnu: {engine}")


def _summarize_backend_result(
    pdf_path: str,
    engine: str,
    result: ExtractionResult,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    type_counts = Counter(block.type for block in result.blocks)
    reader_blocks = result.to_reader_blocks()
    markdown_path = _write_backend_markdown(
        pdf_path,
        engine,
        reader_blocks,
        output_dir=output_dir,
    )

    return {
        "available": True,
        "engine_name": result.engine_name,
        "document_type": result.metadata.get("document_type"),
        "score": result.score,
        "quality_acceptable": _result_quality_acceptable(result),
        "warnings": list(result.warnings),
        "blocks": len(result.blocks),
        "displayable_blocks": sum(1 for block in result.blocks if block.metadata.get("displayable")),
        "classic_routes": sum(1 for block in result.blocks if block.metadata.get("generation_mode") == "classic"),
        "llm_text_routes": sum(
            1 for block in result.blocks if block.metadata.get("generation_mode") == "llm_text_or_multimodal"
        ),
        "llm_multimodal_routes": sum(
            1 for block in result.blocks if block.metadata.get("generation_mode") == "llm_multimodal"
        ),
        "types": dict(type_counts),
        "pages": result.pages,
        "pages_covered": sorted(_covered_pages(result)),
        "formulas": type_counts.get("formula", 0),
        "tables": type_counts.get("table", 0),
        "figures": type_counts.get("figure", 0),
        "assets_missing": blocks_have_missing_managed_assets(reader_blocks),
        "markdown_path": str(markdown_path),
        "debug_paths": list(result.debug_paths),
    }


def _write_backend_markdown(
    pdf_path: str,
    engine: str,
    blocks: list[dict[str, Any]],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    base = Path(output_dir) if output_dir else Path(pdf_path).with_suffix("").parent / "debug_outputs"
    base.mkdir(parents=True, exist_ok=True)

    path = base / f"{Path(pdf_path).stem}_{engine}.md"
    path.write_text(_blocks_to_markdown(blocks), encoding="utf-8")

    return path


def _blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for block in blocks:
        btype = block.get("type")
        text = block.get("text") or block.get("caption") or ""

        if btype == "heading":
            level = min(max(int(block.get("level") or 1), 1), 6)
            lines.append(f"{'#' * level} {text}".strip())

        elif btype == "formula":
            latex = block.get("latex") or text
            lines.append(f"$$\n{latex}\n$$")

        elif btype == "bullet_list":
            items = block.get("items") or []
            lines.extend(f"- {item}" for item in items)

        elif btype == "table":
            lines.append(str(block.get("markdown") or text))

        elif btype == "figure":
            caption = block.get("caption") or text
            image_path = block.get("image_path") or ""
            lines.append(f"![{caption}]({image_path})")

        elif text:
            lines.append(str(text))

        lines.append("")

    return "\n".join(lines).strip() + "\n"
