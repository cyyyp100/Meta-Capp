"""LLM-assisted formula/figure crop refinement per PDF page.

Flow per page:
  1. Render page at 72 DPI → base64 PNG (1 pt ≈ 1 px, same coordinate system)
  2. Send image + compact block JSON to multimodal LLM
  3. LLM returns:
       - refined_crops : [{block_id, bbox [x0,y0,x1,y1], confidence}]
       - missed_formulas: [{bbox, type, confidence}]
  4. For each confirmed crop (confidence ≥ threshold):
       - crop at high DPI (3×) → update block.metadata["llm_crop_path"]
       - set render_mode="pdf_crop" if not already set
  5. Fallback: any error or low confidence → leave blocks unchanged.

Coordinate system: all bboxes are in PDF points (origin = top-left).
At 72 DPI, 1 PDF point = 1 pixel, so no conversion is needed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from document.models import BoundingBox, DocumentBlock
from document.postprocess.figure_extractor import document_asset_dir

logger = logging.getLogger("Document.llm_page_cropper")

_RENDER_DPI = 72          # low-res render sent to LLM (1 pt ≈ 1 px)
_CROP_ZOOM = 3.0          # high-res crop factor for final display
_MIN_CONFIDENCE = 0.60
_FORMULA_TYPES = {"formula", "paragraph", "text", "definition", "theorem", "example", "remark"}
_VISUAL_TYPES = {"figure", "table", "formula"}
_PROMPT = """\
Tu es un analyseur de mise en page de document PDF scientifique.
Je te montre une page complète du PDF (résolution basse, environ 72 DPI).

Coordonnées : les bboxes ci-dessous et dans ta réponse sont en **points PDF** \
(origine coin haut-gauche). À 72 DPI, 1 point ≈ 1 pixel.
Dimensions de la page : {width:.0f} × {height:.0f} points.

Blocs déjà extraits sur cette page :
{blocks_json}

Ta mission (SEULEMENT du JSON, rien d'autre) :
1. "zones_affinées" : pour les blocs existants dont la zone de découpe gagnerait à être ajustée \
(figure ou formule coupée, bords trop serrés, zone trop large, légende textuelle incluse par erreur, etc.), propose une bbox corrigée.  \
Format : [{{"id":"...","bbox":[x0,y0,x1,y1],"confiance":0.0-1.0}}]
2. "formules_manquées" : formules ou équations visibles sur la page qui NE sont PAS dans la liste \
ci-dessus. Format : [{{"bbox":[x0,y0,x1,y1],"type":"formula","confiance":0.0-1.0}}]

Règles strictes :
- N'invente aucun texte, aucun ID, aucun chemin de fichier.
- Retourne uniquement des IDs présents dans la liste ci-dessus pour "zones_affinées".
- Pour une figure, garde les axes, labels, légendes internes et courbes visibles, mais exclue la caption textuelle sous/au-dessus de l'image.
- Si rien à corriger, retourne {{"zones_affinées":[],"formules_manquées":[]}}.
- Confiance < {min_confidence} → ne pas inclure dans la réponse.
"""


def llm_crop_page_formulas(
    pdf_path: str,
    blocks: list[DocumentBlock],
    output_dir: str | Path | None = None,
    model: str | None = None,
    max_pages: int = 20,
    min_confidence: float = _MIN_CONFIDENCE,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    """Use the LLM to refine formula crop regions and detect missed formulas.

    Runs only on pages that contain formula/math content. Returns the same
    block list with updated metadata for blocks whose crop was improved.
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.debug("PyMuPDF indisponible — llm_crop_page_formulas ignoré.")
        return blocks

    from config.settings import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT

    _model = model or OLLAMA_MODEL
    path = Path(pdf_path)
    out_root = Path(output_dir) if output_dir else document_asset_dir(path) / "llm_crops"

    by_page: dict[int, list[DocumentBlock]] = defaultdict(list)
    for block in blocks:
        if block.page is not None:
            by_page[int(block.page)].append(block)

    processed = 0
    with fitz.open(path) as doc:
        for page_num in sorted(by_page):
            if pages is not None and page_num not in pages:
                continue
            if processed >= max_pages:
                break
            page_blocks = by_page[page_num]
            if not _page_needs_llm_crops(page_blocks):
                continue

            page_index = page_num - 1
            if page_index < 0 or page_index >= len(doc):
                continue

            page = doc[page_index]
            page_image_b64 = _render_page_b64(page, dpi=_RENDER_DPI)
            if not page_image_b64:
                continue

            blocks_json = _compact_blocks_json(page_blocks)
            prompt = _PROMPT.format(
                width=page.rect.width,
                height=page.rect.height,
                blocks_json=blocks_json,
                min_confidence=min_confidence,
            )

            try:
                raw = _call_ollama(prompt, _model, [page_image_b64], OLLAMA_URL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT)
                response = _parse_llm_response(raw)
            except Exception as exc:
                logger.debug("LLM page crop page=%d échoué: %s", page_num, exc)
                continue

            id_to_block = {b.id: b for b in page_blocks if b.id}

            # Apply refined crops for existing blocks
            out_root.mkdir(parents=True, exist_ok=True)
            for item in response.get("zones_affinées") or []:
                bid = str(item.get("id") or "")
                confidence = float(item.get("confiance") or 0.0)
                bbox_raw = item.get("bbox")
                if not bid or bid not in id_to_block or not bbox_raw or confidence < min_confidence:
                    continue
                block = id_to_block[bid]
                refined_bbox = _parse_bbox(bbox_raw, page)
                if refined_bbox is None:
                    continue
                crop_path = _do_crop(doc, page_index, refined_bbox, out_root, f"llm_p{page_num}_{bid}")
                if crop_path:
                    logger.info(
                        "[LLM-CROP] page=%d bloc=%s → crop affiné confidence=%.2f → %s",
                        page_num, bid, confidence, crop_path.name,
                    )
                    block.metadata["llm_crop_path"] = str(crop_path)
                    block.metadata["llm_crop_confidence"] = confidence
                    block.metadata["llm_crop_bbox"] = list(refined_bbox)
                    if block.type == "figure":
                        block.image_path = str(crop_path)
                        block.metadata["refined_image_path"] = str(crop_path)
                        block.metadata["figure_crop_refined"] = True
                    elif block.type == "formula":
                        block.metadata["render_mode"] = "pdf_crop"
                        block.image_path = str(crop_path)
                        block.metadata["formula_image_path"] = str(crop_path)
                    elif block.type != "formula":
                        block.metadata.setdefault("context_asset_path", str(crop_path))
                        block.metadata.setdefault("context_asset_type", "llm_crop")
                        block.metadata.setdefault("context_asset_reason", "llm_detected_formula")
                        block.metadata["llm_assets"] = [{
                            "type": "image",
                            "path": str(crop_path),
                            "reason": "llm_detected_formula",
                        }]

            # Insert missed formulas as new blocks to be appended
            for item in response.get("formules_manquées") or []:
                confidence = float(item.get("confiance") or 0.0)
                bbox_raw = item.get("bbox")
                if not bbox_raw or confidence < min_confidence:
                    continue
                refined_bbox = _parse_bbox(bbox_raw, page)
                if refined_bbox is None:
                    continue
                crop_path = _do_crop(
                    doc, page_index, refined_bbox, out_root,
                    f"llm_missed_p{page_num}_{hashlib.md5(str(refined_bbox).encode()).hexdigest()[:6]}",
                )
                if crop_path:
                    logger.info(
                        "[LLM-CROP] page=%d formule manquée détectée confidence=%.2f → %s",
                        page_num, confidence, crop_path.name,
                    )
                    new_block = DocumentBlock(
                        type="formula",
                        text="",
                        page=page_num,
                        bbox=BoundingBox(*refined_bbox),
                        image_path=str(crop_path),
                        confidence=confidence,
                        metadata={
                            "render_mode": "pdf_crop",
                            "formula_image_path": str(crop_path),
                            "llm_crop_path": str(crop_path),
                            "llm_crop_confidence": confidence,
                            "llm_detected_missed": True,
                            "page_width": float(page.rect.width),
                            "page_height": float(page.rect.height),
                        },
                    )
                    blocks.append(new_block)

            processed += 1

    return blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_needs_llm_crops(page_blocks: list[DocumentBlock]) -> bool:
    """Only process pages that have visual or math content worth refining."""
    for b in page_blocks:
        meta = b.metadata or {}
        crop_risk = meta.get("crop_risk") if isinstance(meta.get("crop_risk"), dict) else {}
        if b.type in {"figure", "table"} and (
            crop_risk.get("needs_llm")
            or meta.get("contains_schema")
            or "vector" in str(meta.get("source") or meta.get("geometry_source") or "").casefold()
        ):
            return True
        if b.type == "formula":
            return True
        if meta.get("formula_mode") or meta.get("contains_inline_math"):
            return True
        if meta.get("context_asset_reason") in {"fragmented_math_text", "inline_math", "math_dense_text"}:
            return True
    return False


def _compact_blocks_json(blocks: list[DocumentBlock]) -> str:
    items = []
    for b in blocks:
        meta = b.metadata or {}
        items.append({
            "id": b.id,
            "type": b.type,
            "bbox": b.bbox.to_list() if b.bbox else None,
            "render_mode": meta.get("render_mode"),
            "text_preview": (b.caption or b.text or b.latex or "")[:80],
            "has_image": bool(b.image_path or meta.get("formula_image_path") or meta.get("context_asset_path")),
            "crop_risk": meta.get("crop_risk"),
        })
    return json.dumps(items, ensure_ascii=False)


def _render_page_b64(page: Any, dpi: int = 72) -> str | None:
    """Render the page to a PNG and encode as base64."""
    try:
        zoom = dpi / 72.0
        import fitz  # type: ignore
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        data = pix.tobytes("png")
        return base64.b64encode(data).decode("ascii")
    except Exception as exc:
        logger.debug("Rendu page échoué: %s", exc)
        return None


def _parse_bbox(
    raw: Any,
    page: Any,
) -> tuple[float, float, float, float] | None:
    """Parse and clamp a bbox to the page rect."""
    try:
        import fitz  # type: ignore
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in raw[:4])
        # Clamp to page
        x0 = max(x0, page.rect.x0)
        y0 = max(y0, page.rect.y0)
        x1 = min(x1, page.rect.x1)
        y1 = min(y1, page.rect.y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (x0, y0, x1, y1)
    except (TypeError, ValueError):
        return None


def _do_crop(
    doc: Any,
    page_index: int,
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    name_stem: str,
) -> Path | None:
    """Crop a page region at high resolution and save as PNG."""
    try:
        import fitz  # type: ignore
        page = doc[page_index]
        rect = fitz.Rect(*bbox)
        # Small padding
        rect = fitz.Rect(rect.x0 - 10, rect.y0 - 8, rect.x1 + 10, rect.y1 + 8)
        rect = fitz.Rect(
            max(rect.x0, page.rect.x0), max(rect.y0, page.rect.y0),
            min(rect.x1, page.rect.x1), min(rect.y1, page.rect.y1),
        )
        if rect.is_empty or rect.width < 4 or rect.height < 4:
            return None
        digest = hashlib.md5(str(rect).encode()).hexdigest()[:8]
        out_path = out_dir / f"{name_stem}_{digest}.png"
        if not out_path.exists():
            pix = page.get_pixmap(matrix=fitz.Matrix(_CROP_ZOOM, _CROP_ZOOM), clip=rect, alpha=False)
            pix.save(str(out_path))
        return out_path
    except Exception as exc:
        logger.debug("Crop high-res échoué: %s", exc)
        return None


def _parse_llm_response(raw: str) -> dict[str, list]:
    """Parse LLM JSON, tolerating wrapped responses."""
    for attempt in [raw, raw[raw.find("{"):raw.rfind("}") + 1] if "{" in raw else ""]:
        try:
            data = json.loads(attempt)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    logger.debug("LLM page crop: réponse non parseable: %.120s", raw)
    return {}


def _call_ollama(
    prompt: str,
    model: str,
    images: list[str],
    url: str,
    keep_alive: str,
    timeout: int,
) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "images": images,
        "keep_alive": keep_alive,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")
        return str(data.get("response", ""))
