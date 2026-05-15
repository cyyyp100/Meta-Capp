# core/parser.py - Interface commune d'extraction PDF (cascade moteurs)
from __future__ import annotations

import importlib.util
import logging
import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("Parser")

STRUCTURED_ENGINE = "pymupdf_structured"

# ---------------------------------------------------------------------------
# Mots-clés pour la détection automatique de matière
# ---------------------------------------------------------------------------
_SUBJECT_KEYWORDS: dict[str, frozenset[str]] = {
    "mathématiques": frozenset([
        "mathématiques", "mathematics", "maths", "math",
        "algèbre", "algebra", "calcul", "calculus",
        "analyse", "analysis", "géométrie", "geometry",
        "statistiques", "statistics", "probabilités", "probability",
        "équations", "trigonométrie", "trigonometry", "arithmétique",
        "théorème", "theorem", "intégrale", "integral",
        "dérivée", "derivative", "matrice", "matrix",
        "vecteur", "vector", "démonstration", "lemme",
    ]),
    "sciences": frozenset([
        "physique", "chimie", "biologie", "sciences",
        "physics", "chemistry", "biology", "science",
        "mécanique", "électricité", "thermodynamique",
        "atome", "molécule", "énergie", "force",
    ]),
    "histoire": frozenset([
        "histoire", "history", "historical", "historique",
        "guerre", "révolution", "empire", "siècle",
        "civilisation", "moyen âge", "antiquité",
    ]),
    "géographie": frozenset([
        "géographie", "geography", "géographique",
        "continent", "territoire", "population", "climat",
        "carte", "relief", "frontière",
    ]),
    "français": frozenset([
        "français", "littérature", "grammaire", "orthographe",
        "poésie", "roman", "syntaxe", "conjugaison",
        "vocabulaire", "rédaction", "dissertation",
    ]),
    "informatique": frozenset([
        "informatique", "programmation", "algorithmique",
        "computer", "algorithm", "python", "javascript",
        "programming", "code", "logiciel", "réseau",
        "données", "base de données", "complexité",
    ]),
}
ENGINE_CASCADE = ("pymupdf",)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+")
_CAPTION_RE = re.compile(
    r"^(Figure|Fig\.?|Schema|Schéma|Diagramme|Tableau|Table)\s+"
    r"[\divxlcdmIVXLCDM]+(?:\s*[:.-]\s*|\s+).+"
)
_INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
_WHOLE_INLINE_FORMULA_RE = re.compile(r"^\$([^$\n]+?)\$$", re.DOTALL)
# ---------------------------------------------------------------------------
# Interface commune : chaque moteur retourne List[Block] pour une page
# Block = {"type": str, ...} selon spec §10.5
# ---------------------------------------------------------------------------

def extract_page(pdf_path: str, page_number: int, engine: str = "pymupdf_structured") -> list[dict]:
    """Extrait une page via le pipeline PyMuPDF structuré."""
    if page_number < 1:
        raise ValueError("page_number doit être 1-based et >= 1")

    extractors = {
        STRUCTURED_ENGINE: _extract_pymupdf_structured,
        "pymupdf": _extract_pymupdf,
    }

    errors: list[str] = []
    for current_engine in _fallback_chain(engine):
        if current_engine != "pymupdf" and not _engine_available(current_engine):
            logger.warning("[%s] indisponible -> fallback", current_engine.upper())
            continue

        try:
            blocks = extractors[current_engine](pdf_path, page_number)
            if not blocks and current_engine != "pymupdf":
                logger.warning("[%s] p.%s sans blocs -> fallback", current_engine.upper(), page_number)
                continue
            _tag_blocks(blocks, page_number, current_engine)
            logger.debug(
                "[%s] p.%s -> %s blocs",
                current_engine.upper(),
                page_number,
                len(blocks),
            )
            return blocks
        except Exception as exc:
            errors.append(f"{current_engine}: {exc}")
            logger.warning(
                "[%s] echec p.%s : %s -> fallback",
                current_engine.upper(),
                page_number,
                exc,
            )

    raise RuntimeError("Aucun moteur d'extraction disponible: " + " | ".join(errors))


def extract_page_lazy(
    pdf_path: str,
    page_number: int,
    prev_page_tail: list[str] | None = None,
    enrich_assets: bool = True,
    *,
    document_type: str | None = None,
    validate_with_llm: bool = True,
    llm_generation: int | None = None,
):
    try:
        from document.page_pipeline import extract_page_lazy as _extract_page_lazy
    except ModuleNotFoundError:
        from nwol.document.page_pipeline import extract_page_lazy as _extract_page_lazy

    return _extract_page_lazy(
        pdf_path,
        page_number,
        prev_page_tail=prev_page_tail,
        enrich_assets=enrich_assets,
        document_type=document_type,
        validate_with_llm=validate_with_llm,
        llm_generation=llm_generation,
    )


def detect_best_engine(pdf_path: str | None = None) -> str:
    """Détecte le moteur disponible (toujours PyMuPDF structuré)."""
    if pdf_path:
        logger.info("Type documentaire detecte : %s", detect_document_type(pdf_path))

    if _engine_available(STRUCTURED_ENGINE):
        logger.info("Moteur selectionne : %s", STRUCTURED_ENGINE)
        return STRUCTURED_ENGINE

    logger.warning("PyMuPDF (fitz) indisponible")
    return "pymupdf"


def detect_document_type(pdf_path: str) -> str:
    """
    Heuristique legere : scientific si deux colonnes, slides si images dominantes,
    book sinon.
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            if len(doc) == 0:
                return "book"
            page = doc[0]
            text_dict = page.get_text("dict")
            raw_blocks = text_dict.get("blocks", [])
            text_blocks = [b for b in raw_blocks if b.get("type") == 0 and _fitz_block_text(b)]
            image_blocks = [b for b in raw_blocks if b.get("type") == 1]
            text_chars = sum(len(_fitz_block_text(b)) for b in text_blocks)

            if _blocks_are_multicolumn(text_blocks, float(page.rect.width), min_per_column=2):
                return "scientific"
            if image_blocks and (len(image_blocks) >= len(text_blocks) or text_chars < 600):
                return "slides"
            return "book"
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("Detection type documentaire impossible pour %s: %s", pdf_path, exc)
        return "book"


def extract_first_pages_text(pdf_path: str, n: int = 2, max_chars: int = 800) -> str:
    """Extrait le texte brut des n premières pages du PDF pour le LLM."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            parts: list[str] = []
            for i in range(min(n, len(doc))):
                parts.append(doc[i].get_text("text"))
                if sum(len(p) for p in parts) >= max_chars:
                    break
            return "".join(parts)[:max_chars]
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("Extraction texte premières pages impossible pour %s: %s", pdf_path, exc)
        return ""


def detect_document_subject(pdf_path: str) -> str | None:
    """Détecte la matière du document (mathématiques, sciences, etc.) depuis le nom de
    fichier et les premières pages. Retourne None si non identifiable."""
    filename = Path(pdf_path).stem.lower()
    scores: dict[str, int] = {}

    def _score(subject: str, text: str, weight: int) -> None:
        count = sum(1 for kw in _SUBJECT_KEYWORDS[subject] if kw in text)
        if count:
            scores[subject] = scores.get(subject, 0) + count * weight

    for subj in _SUBJECT_KEYWORDS:
        _score(subj, filename, 3)

    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            pages_to_check = min(3, len(doc))
            for i in range(pages_to_check):
                text = doc[i].get_text("text").lower()
                for subj in _SUBJECT_KEYWORDS:
                    _score(subj, text, 1)
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("Détection matière impossible pour %s: %s", pdf_path, exc)

    if not scores:
        return "culture"
    best = max(scores, key=lambda s: scores[s])
    logger.info("Matière détectée : %s (score=%d) pour %s", best, scores[best], pdf_path)
    return best


def _fallback_chain(engine: str) -> tuple[str, ...]:
    if engine == STRUCTURED_ENGINE:
        return (STRUCTURED_ENGINE, *ENGINE_CASCADE)
    if engine in ENGINE_CASCADE:
        start = ENGINE_CASCADE.index(engine)
        return ENGINE_CASCADE[start:]
    return ENGINE_CASCADE


def _engine_available(engine: str) -> bool:
    if engine in (STRUCTURED_ENGINE, "pymupdf"):
        return importlib.util.find_spec("fitz") is not None
    return False


def _tag_blocks(blocks: list[dict], page_number: int, engine: str) -> None:
    for idx, block in enumerate(blocks):
        block.setdefault("page_number", page_number)
        block.setdefault("page_start", page_number)
        block.setdefault("page_end", page_number)
        block.setdefault("engine", engine)
        block.setdefault("block_index", idx)


def get_extraction_report(pdf_path: str, engine: str | None = None) -> dict[str, Any] | None:
    """Retourne les metadonnees du nouveau pipeline, si le moteur l'utilise."""
    selected_engine = engine or detect_best_engine(pdf_path)
    if selected_engine != STRUCTURED_ENGINE:
        return None

    _, score, warnings, stats = _structured_document_payload(str(Path(pdf_path).resolve()))
    return {
        "engine": stats.get("engine_name") or STRUCTURED_ENGINE,
        "score": score,
        "warnings": list(warnings),
        "stats": dict(stats),
    }


def clear_extraction_caches() -> None:
    _structured_document_payload.cache_clear()
    try:
        try:
            from document.pdf_router import clear_cache as clear_document_router_cache
        except ModuleNotFoundError:
            from nwol.document.pdf_router import clear_cache as clear_document_router_cache

        clear_document_router_cache()
    except Exception as exc:
        logger.debug("Cache routeur document non vidé: %s", exc)

    try:
        try:
            from pdf.pipeline import clear_cache as clear_pdf_cache
        except ModuleNotFoundError:
            from nwol.pdf.pipeline import clear_cache as clear_pdf_cache

        clear_pdf_cache()
    except Exception as exc:
        logger.debug("Cache PDF non vidé: %s", exc)


# ---------------------------------------------------------------------------
# Parsing Markdown commun (PyMuPDF4LLM, Marker, MinerU)
# ---------------------------------------------------------------------------

def _parse_markdown_blocks(md_text: str, base_path: str | Path | None = None) -> list[dict]:
    """
    Transforme un Markdown de page en blocs structures.

    Ordre de reconnaissance :
    display math, inline math seul, code fence/indentation, headings, figures,
    captions, paragraphes.
    """
    lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    base = Path(base_path) if base_path else None
    blocks: list[dict] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    i = 0

    def flush_list_items() -> None:
        nonlocal list_items
        if not list_items:
            return
        items = list_items[:]
        list_items = []
        if len(items) == 1:
            text = items[0]
            blocks.append({"type": "paragraph", "text": text})
        else:
            blocks.append({"type": "paragraph", "text": "\n".join(items), "is_list": True})

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = _join_paragraph_lines(paragraph)
        paragraph = []
        if not text:
            return

        display_latex = _whole_display_latex(text)
        if display_latex is not None:
            blocks.append({"type": "formula", "latex": display_latex, "display": True})
            return

        inline_match = _WHOLE_INLINE_FORMULA_RE.match(text)
        if inline_match:
            blocks.append({
                "type": "formula",
                "latex": inline_match.group(1).strip(),
                "display": False,
            })
            return

        block = {"type": "paragraph", "text": text}
        inline_formulas = [m.strip() for m in _INLINE_FORMULA_RE.findall(text) if m.strip()]
        if inline_formulas:
            block["inline_formulas"] = inline_formulas
        if _CAPTION_RE.match(text):
            block["is_caption"] = True
        blocks.append(block)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            # Ne pas flusher les list_items sur ligne vide : les puces sont souvent séparées
            i += 1
            continue

        if stripped.startswith("$$"):
            flush_paragraph()
            formula_lines = [stripped]
            i += 1
            while i < len(lines) and not _display_formula_is_closed(formula_lines):
                formula_lines.append(lines[i].strip())
                i += 1
            latex = _strip_display_formula("\n".join(formula_lines))
            blocks.append({"type": "formula", "latex": latex, "display": True})
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip() or None
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            block = {"type": "code", "text": "\n".join(code_lines).rstrip()}
            if language:
                block["language"] = language
            blocks.append(block)
            continue

        if line.startswith(("    ", "\t")):
            flush_paragraph()
            code_lines = []
            while i < len(lines) and (lines[i].startswith(("    ", "\t")) or not lines[i].strip()):
                code_lines.append(_strip_code_indent(lines[i]))
                i += 1
            blocks.append({"type": "code", "text": "\n".join(code_lines).rstrip()})
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            level = min(len(heading_match.group(1)), 3)
            blocks.append({
                "type": "heading",
                "level": level,
                "text": heading_match.group(2).strip(),
            })
            i += 1
            continue

        image_match = _IMAGE_RE.match(stripped)
        if image_match:
            flush_paragraph()
            image_path = _resolve_markdown_image_path(image_match.group("path"), base)
            blocks.append({
                "type": "figure",
                "image_path": image_path,
                "alt": image_match.group("alt").strip(),
                "caption": "",
            })
            i += 1
            continue

        if _CAPTION_RE.match(stripped):
            flush_paragraph()
            flush_list_items()
            if blocks and blocks[-1].get("type") == "figure":
                blocks[-1]["caption"] = stripped
            else:
                blocks.append({"type": "paragraph", "text": stripped, "is_caption": True})
            i += 1
            continue

        if _LIST_ITEM_RE.match(stripped):
            flush_paragraph()
            list_items.append(stripped)
            i += 1
            continue

        # Ligne de texte normal : si on était dans une liste, la fermer
        flush_list_items()
        paragraph.append(line)
        i += 1

    flush_paragraph()
    flush_list_items()
    return blocks


def _join_paragraph_lines(lines: list[str]) -> str:
    cleaned = [re.sub(r"\s+", " ", line.strip()) for line in lines if line.strip()]
    return " ".join(cleaned).strip()


def _display_formula_is_closed(lines: list[str]) -> bool:
    text = "\n".join(lines).strip()
    return text.endswith("$$") and text.count("$$") >= 2 and len(text) > 2


def _strip_display_formula(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$$"):
        stripped = stripped[2:]
    if stripped.endswith("$$"):
        stripped = stripped[:-2]
    return stripped.strip()


def _whole_display_latex(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
        return stripped[2:-2].strip()
    return None


def _strip_code_indent(line: str) -> str:
    if line.startswith("    "):
        return line[4:]
    if line.startswith("\t"):
        return line[1:]
    return line


def _resolve_markdown_image_path(raw_path: str, base: Path | None) -> str:
    path = raw_path.strip().split()[0].strip("<>")
    if re.match(r"^[a-z]+://", path, re.IGNORECASE):
        return path
    candidate = Path(path)
    if candidate.is_absolute() or base is None:
        return str(candidate)
    return str((base / candidate).resolve())


# ---------------------------------------------------------------------------
# Nouveau moteur reader/PyMuPDF structure
# ---------------------------------------------------------------------------

def _extract_pymupdf_structured(pdf_path: str, page_number: int) -> list[dict]:
    result = extract_page_lazy(
        str(Path(pdf_path).resolve()),
        page_number,
        enrich_assets=True,
    )
    return result.to_reader_blocks()


@lru_cache(maxsize=2)
def _structured_document_payload(pdf_path: str) -> tuple[tuple[dict[str, Any], ...], float, tuple[str, ...], dict[str, Any]]:
    try:
        from pdf.pipeline import build_document_model
    except ModuleNotFoundError:
        from nwol.pdf.pipeline import build_document_model

    result = build_document_model(pdf_path, preferred_engine="auto")
    blocks = result.to_reader_blocks()
    logger.info(
        "[%s] document converti : %s bloc(s), score=%s",
        result.engine_name.upper(),
        len(blocks),
        result.score,
    )
    return (
        tuple(_clone_block(block) for block in blocks),
        result.score,
        tuple(result.warnings),
        {
            "pages": result.pages,
            "blocks": len(blocks),
            "engine_name": result.engine_name,
            "debug_paths": list(result.debug_paths),
        },
    )


def _clone_block(block: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, list):
            cloned[key] = value[:]
        elif isinstance(value, dict):
            cloned[key] = dict(value)
        else:
            cloned[key] = value
    return cloned


# ---------------------------------------------------------------------------
# Moteur PyMuPDF/PyMuPDF4LLM (fallback obligatoire)
# ---------------------------------------------------------------------------

def _extract_pymupdf(pdf_path: str, page_number: int) -> list[dict]:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        dict_blocks = _extract_fitz_dict_blocks(page)
        page_width = float(page.rect.width)

        if _blocks_are_multicolumn(
            [b for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0],
            page_width,
            min_per_column=2,
        ):
            return dict_blocks

        markdown = _pymupdf4llm_markdown(pdf_path, page_number, doc)
        md_blocks = _parse_markdown_blocks(markdown) if markdown.strip() else []

        if _prefer_dict_blocks(dict_blocks, md_blocks):
            return dict_blocks
        return md_blocks
    finally:
        doc.close()


def _pymupdf4llm_markdown(pdf_path: str, page_number: int, doc: Any) -> str:
    try:
        import pymupdf4llm
    except ImportError:
        return ""

    page_index = page_number - 1
    try:
        rendered = pymupdf4llm.to_markdown(pdf_path, pages=[page_index])
    except TypeError:
        rendered = pymupdf4llm.to_markdown(doc, pages=[page_index])
    except Exception as exc:
        logger.debug("PyMuPDF4LLM indisponible sur p.%s: %s", page_number, exc)
        return ""

    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, list):
        parts = []
        for item in rendered:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("markdown") or ""))
            else:
                parts.append(str(item))
        return "\n\n".join(part for part in parts if part.strip())
    return str(rendered)


def _extract_fitz_dict_blocks(page: Any) -> list[dict]:
    raw_blocks = page.get_text("dict").get("blocks", [])
    text_blocks = [b for b in raw_blocks if b.get("type") == 0 and _fitz_block_text(b)]
    font_sizes = [
        span.get("size", 12)
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    body_size = statistics.median(font_sizes) if font_sizes else 12.0

    blocks: list[dict] = []
    for raw in _sort_pdf_blocks_for_reading(raw_blocks, float(page.rect.width)):
        if raw.get("type") == 0:
            text = _fitz_block_text(raw)
            if not text:
                continue
            blocks.append(_classify_fitz_text_block(raw, text, body_size))
        elif raw.get("type") == 1:
            blocks.append({
                "type": "figure",
                "image_path": None,
                "bbox": raw.get("bbox", []),
                "caption": "",
            })
    return blocks


def _fitz_block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = sorted(line.get("spans", []), key=lambda span: span.get("bbox", [0])[0])
        line_text = "".join(span.get("text", "") for span in spans).strip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def _classify_fitz_text_block(block: dict, text: str, body_size: float) -> dict:
    display_latex = _whole_display_latex(text)
    if display_latex is not None:
        return {"type": "formula", "latex": display_latex, "display": True, "bbox": block.get("bbox", [])}

    inline_match = _WHOLE_INLINE_FORMULA_RE.match(text)
    if inline_match:
        return {
            "type": "formula",
            "latex": inline_match.group(1).strip(),
            "display": False,
            "bbox": block.get("bbox", []),
        }

    font_sizes = [
        span.get("size", body_size)
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else body_size
    max_size = max(font_sizes) if font_sizes else body_size

    if avg_size >= max(16.0, body_size * 1.35) or max_size >= max(18.0, body_size * 1.5):
        return {"type": "heading", "level": 1, "text": _flatten_text(text), "bbox": block.get("bbox", [])}
    if avg_size >= max(13.0, body_size * 1.18) or _looks_like_numbered_heading(text):
        return {"type": "heading", "level": 2, "text": _flatten_text(text), "bbox": block.get("bbox", [])}

    paragraph = {"type": "paragraph", "text": _flatten_text(text), "bbox": block.get("bbox", [])}
    inline_formulas = [m.strip() for m in _INLINE_FORMULA_RE.findall(text) if m.strip()]
    if inline_formulas:
        paragraph["inline_formulas"] = inline_formulas
    if _CAPTION_RE.match(paragraph["text"]):
        paragraph["is_caption"] = True
    return paragraph


def _flatten_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_numbered_heading(text: str) -> bool:
    flattened = _flatten_text(text)
    return bool(re.match(r"^\d+(?:\.\d+){0,3}\s+[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", flattened))


def _prefer_dict_blocks(dict_blocks: list[dict], md_blocks: list[dict]) -> bool:
    if not md_blocks:
        return True
    if len(md_blocks) == 1 and len(dict_blocks) > 1:
        return True
    if any(b.get("type") == "heading" for b in dict_blocks) and not any(
        b.get("type") == "heading" for b in md_blocks
    ):
        return True
    return False


def _sort_pdf_blocks_for_reading(blocks: list[dict], page_width: float | None = None) -> list[dict]:
    """
    Trie les blocs PDF en ordre de lecture.

    En page mono-colonne : haut -> bas, puis gauche -> droite.
    En page multi-colonnes : colonne gauche complete, puis colonne droite.
    """
    blocks_with_bbox = [b for b in blocks if _valid_bbox(b.get("bbox"))]
    if not blocks_with_bbox:
        return blocks

    width = page_width or max(float(b["bbox"][2]) for b in blocks_with_bbox)
    multicolumn = _blocks_are_multicolumn(blocks_with_bbox, width, min_per_column=1)
    column_candidates = [
        b for b in blocks_with_bbox
        if (float(b["bbox"][2]) - float(b["bbox"][0])) < width * 0.72
    ]
    column_top = min((float(b["bbox"][1]) for b in column_candidates), default=0.0)
    column_bottom = max((float(b["bbox"][3]) for b in column_candidates), default=0.0)

    def sort_key(block: dict) -> tuple[float, float, float]:
        x0, y0, x1, _ = [float(v) for v in block["bbox"][:4]]
        if not multicolumn:
            return (y0, x0, 0.0)
        block_width = x1 - x0
        if block_width >= width * 0.72:
            y1 = float(block["bbox"][3])
            if y1 <= column_top:
                return (-1.0, y0, x0)
            if y0 >= column_bottom:
                return (2.0, y0, x0)
            return (0.5, y0, x0)
        center = (x0 + x1) / 2
        column = 0.0 if center < width / 2 else 1.0
        return (column, y0, x0)

    return sorted(blocks, key=lambda block: sort_key(block) if _valid_bbox(block.get("bbox")) else (9999, 9999, 0))


def _blocks_are_multicolumn(
    blocks: list[dict],
    page_width: float,
    min_per_column: int = 2,
) -> bool:
    candidates = [
        b for b in blocks
        if _valid_bbox(b.get("bbox"))
        and (float(b["bbox"][2]) - float(b["bbox"][0])) < page_width * 0.72
    ]
    if len(candidates) < min_per_column * 2:
        return False

    left = [b for b in candidates if _bbox_center_x(b["bbox"]) < page_width * 0.48]
    right = [b for b in candidates if _bbox_center_x(b["bbox"]) > page_width * 0.52]
    if len(left) < min_per_column or len(right) < min_per_column:
        return False

    left_range = _vertical_range(left)
    right_range = _vertical_range(right)
    overlap = min(left_range[1], right_range[1]) - max(left_range[0], right_range[0])
    return overlap > 0


def _valid_bbox(bbox: Any) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) >= 4


def _bbox_center_x(bbox: list | tuple) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2


def _vertical_range(blocks: list[dict]) -> tuple[float, float]:
    return (
        min(float(b["bbox"][1]) for b in blocks),
        max(float(b["bbox"][3]) for b in blocks),
    )
