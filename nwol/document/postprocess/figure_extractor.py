from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from config.settings import ASSETS_DIR
from document.models import BoundingBox, DocumentBlock

logger = logging.getLogger("Document.figure_extractor")

CAPTION_RE = re.compile(
    r"^(Figure|Fig\.?|Schema|Schéma|Graphique|Diagramme|Illustration|Tableau|Table)\s*\d*\.?\b",
    re.I,
)


def extract_native_figures(
    pdf_path: str,
    output_root: str | Path | None = None,
    *,
    pages: set[int] | None = None,
) -> list[DocumentBlock]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        logger.debug("PyMuPDF indisponible pour extraction figures: %s", exc)
        return []

    path = Path(pdf_path)
    output_dir = document_asset_dir(path, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[DocumentBlock] = []
    seen_occurrences: set[tuple[int, int, tuple[int, int, int, int]]] = set()

    try:
        with fitz.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                if pages is not None and page_index not in pages:
                    continue
                img_index = 0
                for image in page.get_images(full=True):
                    xref = image[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    img_index += 1
                    for rect in rects[:1]:
                        bbox = BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                        if bbox.width < 32 or bbox.height < 32:
                            continue
                        aspect = bbox.width / max(bbox.height, 1)
                        if aspect > 12.0 or aspect < 1 / 12.0:
                            continue

                        occurrence = (page_index, int(xref), _bbox_signature(bbox))
                        if occurrence in seen_occurrences:
                            continue
                        seen_occurrences.add(occurrence)

                        _CROP_VERSION = "v4"
                        digest = hashlib.md5(
                            f"{path}-{page_index}-{xref}-{bbox.to_list()}-{_CROP_VERSION}".encode(
                                "utf-8",
                                errors="ignore",
                            )
                        ).hexdigest()[:10]
                        image_path = output_dir / f"page_{page_index}_img_{img_index}_{digest}.png"
                        try:
                            if not image_path.exists():
                                clip = fitz.Rect(*bbox.to_list())
                                pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(3, 3), alpha=False)
                                if not _save_trimmed_figure_pixmap(pix, image_path):
                                    pix.save(str(image_path))
                            image_path_str = str(image_path)
                        except Exception as exc:
                            logger.debug("Image PDF ignorée p.%s xref=%s: %s", page_index, xref, exc)
                            continue
                        if not image_path_str:
                            continue
                        figures.append(
                            DocumentBlock(
                                type="figure",
                                image_path=image_path_str,
                                caption="",
                                text="",
                                page=page_index,
                                bbox=bbox,
                                confidence=0.85,
                                metadata={"xref": xref, "source": "pdf_native_image"},
                            )
                        )
    except Exception as exc:
        logger.warning("Extraction des images natives échouée: %s", exc)
    return deduplicate_visual_blocks(figures)


def _save_trimmed_figure_pixmap(pix, image_path: Path) -> bool:
    try:
        from PIL import Image
        mode = "RGBA" if getattr(pix, "alpha", False) else "RGB"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        image = _trim_light_background_figure(image, padding=4)
        image.save(image_path)
        return True
    except Exception as exc:
        logger.debug("Rognage figure ignoré pour %s: %s", image_path, exc)
        return False


def _trim_light_background_figure(image, *, padding: int = 16):
    try:
        gray = image.convert("L")
        mask = gray.point(lambda pixel: 255 if pixel < 248 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return image
        left, top, right, bottom = bbox
        if right - left < 2 or bottom - top < 2:
            return image
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)
        if left == 0 and top == 0 and right == image.width and bottom == image.height:
            return image
        return image.crop((left, top, right, bottom))
    except Exception:
        return image


def deduplicate_visual_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Remove duplicate figure/crop blocks while preserving the richest metadata.

    Complex PDFs often expose the same visual region twice: once through a
    structured backend and once through the native image/crop pass. We only
    collapse near-identical visual regions, not subfigures inside a larger
    composite figure.
    """
    result: list[DocumentBlock] = []
    image_hash_cache: dict[str, str | None] = {}

    for block in blocks:
        if block.type != "figure":
            result.append(block)
            continue

        duplicate_index = None
        for index, existing in enumerate(result):
            if existing.type != "figure":
                continue
            if _figures_are_duplicates(existing, block, image_hash_cache):
                duplicate_index = index
                break

        if duplicate_index is None:
            result.append(block)
            continue

        result[duplicate_index] = _merge_duplicate_figures(result[duplicate_index], block)

    return result


def document_asset_dir(pdf_path: str | Path, output_root: str | Path | None = None) -> Path:
    return _asset_root(output_root) / _document_id(Path(pdf_path))


def cleanup_document_assets(pdf_path: str | Path, output_root: str | Path | None = None) -> int:
    output_dir = document_asset_dir(pdf_path, output_root=output_root).resolve()
    root = _asset_root(output_root).resolve()
    if output_dir == root or root not in output_dir.parents:
        logger.warning("Nettoyage assets ignoré, chemin non sûr: %s", output_dir)
        return 0
    if not output_dir.exists():
        return 0

    file_count = sum(1 for path in output_dir.rglob("*") if path.is_file())
    shutil.rmtree(output_dir)
    logger.info("Assets temporaires supprimés: %s fichier(s) dans %s", file_count, output_dir)
    return file_count


def cleanup_all_document_assets(output_root: str | Path | None = None) -> int:
    root = _asset_root(output_root).resolve()
    if not root.exists():
        return 0
    if not root.is_dir():
        logger.warning("Nettoyage assets ignoré, racine invalide: %s", root)
        return 0

    file_count = sum(1 for path in root.rglob("*") if path.is_file())
    for child in root.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            logger.warning("Suppression asset échouée %s: %s", child, exc)
    logger.info("Cache assets temporaire supprimé: %s fichier(s) dans %s", file_count, root)
    return file_count


def blocks_have_missing_managed_assets(blocks: list[dict]) -> bool:
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        for image_path in _block_asset_paths(block):
            if image_path and _is_managed_asset_path(image_path) and not _asset_path_exists(image_path):
                return True
    return False


def _block_asset_paths(block: dict) -> list[str]:
    paths: list[str] = []
    image_path = block.get("image_path")
    if image_path:
        paths.append(str(image_path))

    metadata = block.get("metadata") or {}
    for key in ("formula_image_path", "context_asset_path", "table_image_path"):
        path = metadata.get(key)
        if path:
            paths.append(str(path))
    for asset in metadata.get("llm_assets") or []:
        if isinstance(asset, dict) and asset.get("path"):
            paths.append(str(asset["path"]))
    return paths


def _asset_root(output_root: str | Path | None = None) -> Path:
    root = Path(output_root or ASSETS_DIR)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def _is_managed_asset_path(path: str | Path) -> bool:
    raw = Path(path)
    parts = raw.parts
    if "assets" in parts:
        return True
    root = _asset_root().resolve()
    for candidate in _asset_path_candidates(raw):
        try:
            if candidate.resolve().is_relative_to(root):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def _asset_path_exists(path: str | Path) -> bool:
    return any(candidate.exists() for candidate in _asset_path_candidates(Path(path)))


def _asset_path_candidates(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]

    candidates = [Path.cwd() / path]
    parts = path.parts
    if parts and parts[0] == "assets":
        candidates.append(_asset_root() / Path(*parts[1:]))
    if len(parts) >= 2 and parts[0] == "nwol" and parts[1] == "assets":
        candidates.append(_asset_root() / Path(*parts[2:]))
    return candidates


def associate_captions(
    text_blocks: list[DocumentBlock],
    figure_blocks: list[DocumentBlock],
    max_distance: float = 180.0,
) -> list[DocumentBlock]:
    if not figure_blocks:
        return text_blocks

    captions = [
        block
        for block in text_blocks
        if block.type == "paragraph" and (CAPTION_RE.match(block.text.strip()) or block.metadata.get("is_caption"))
    ]
    assigned_figure_ids: set[int] = set()
    used_caption_ids: set[int] = set()
    for caption_index, caption in enumerate(captions, start=1):
        group = _caption_figure_group(caption, figure_blocks, assigned_figure_ids, max_distance=max_distance)
        if not group:
            continue
        group_id = f"p{caption.page or 0}_caption_{caption_index}"
        for display_index, figure in enumerate(group):
            _attach_caption(figure, caption.text.strip())
            figure.metadata["caption_group"] = group_id
            figure.metadata["caption_display"] = display_index == 0
            assigned_figure_ids.add(id(figure))
        used_caption_ids.add(id(caption))

    for figure in figure_blocks:
        if id(figure) in assigned_figure_ids:
            continue
        nearest = _nearest_caption(figure, captions, used_caption_ids, max_distance=max_distance)
        if nearest is None:
            continue
        _attach_caption(figure, nearest.text.strip())
        figure.metadata["caption_display"] = True
        used_caption_ids.add(id(nearest))

    merged = [block for block in text_blocks if id(block) not in used_caption_ids]
    merged.extend(figure_blocks)
    return sorted(merged, key=_position_key)


def _caption_figure_group(
    caption: DocumentBlock,
    figures: list[DocumentBlock],
    assigned: set[int],
    max_distance: float,
) -> list[DocumentBlock]:
    if caption.bbox is None:
        return []

    above: list[tuple[float, DocumentBlock]] = []
    below: list[tuple[float, DocumentBlock]] = []
    for figure in figures:
        if id(figure) in assigned or figure.page != caption.page or figure.bbox is None:
            continue
        if not _caption_matches_visual(caption, figure):
            continue
        if not _horizontally_related(figure.bbox, caption.bbox):
            continue
        if figure.bbox.y1 <= caption.bbox.y0:
            distance = caption.bbox.y0 - figure.bbox.y1
            if 0 <= distance <= max_distance:
                above.append((distance, figure))
        elif figure.bbox.y0 >= caption.bbox.y1:
            distance = figure.bbox.y0 - caption.bbox.y1
            if 0 <= distance <= max_distance:
                below.append((distance, figure))

    group: list[DocumentBlock] = []
    if above:
        group.append(min(above, key=lambda item: item[0])[1])
    if below:
        nearest_below = min(below, key=lambda item: item[0])[1]
        if id(nearest_below) not in {id(item) for item in group}:
            group.append(nearest_below)
    return sorted(group, key=_position_key)


def _nearest_caption(
    figure: DocumentBlock,
    captions: list[DocumentBlock],
    used: set[int],
    max_distance: float,
) -> DocumentBlock | None:
    if figure.bbox is None:
        return None
    candidates: list[tuple[float, DocumentBlock]] = []
    for caption in captions:
        if id(caption) in used or caption.page != figure.page or caption.bbox is None:
            continue
        if not _caption_matches_visual(caption, figure):
            continue
        if not _horizontally_related(figure.bbox, caption.bbox):
            continue
        if caption.bbox.y0 >= figure.bbox.y1:
            distance = caption.bbox.y0 - figure.bbox.y1
        else:
            distance = figure.bbox.y0 - caption.bbox.y1
        if 0 <= distance <= max_distance:
            candidates.append((distance, caption))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _attach_caption(block: DocumentBlock, caption_text: str) -> None:
    block.caption = caption_text
    if block.type == "figure":
        block.text = caption_text


def _caption_matches_visual(caption: DocumentBlock, visual: DocumentBlock) -> bool:
    kind = _caption_kind(caption.text)
    if kind == "table":
        return visual.type == "table"
    if kind == "figure":
        return visual.type == "figure"
    return visual.type in {"figure", "table"}


def _caption_kind(text: str) -> str | None:
    match = CAPTION_RE.match(str(text or "").strip())
    if not match:
        return None
    label = match.group(1).casefold()
    if label in {"table", "tableau"}:
        return "table"
    return "figure"


def _horizontally_related(figure_bbox: BoundingBox, caption_bbox: BoundingBox) -> bool:
    overlap = min(figure_bbox.x1, caption_bbox.x1) - max(figure_bbox.x0, caption_bbox.x0)
    if overlap > 0:
        return True
    caption_center = caption_bbox.center_x
    figure_center = figure_bbox.center_x
    return abs(caption_center - figure_center) <= max(figure_bbox.width, caption_bbox.width) * 0.6


def _figures_are_duplicates(
    left: DocumentBlock,
    right: DocumentBlock,
    image_hash_cache: dict[str, str | None],
) -> bool:
    left_path = _normalized_image_path(left.image_path)
    right_path = _normalized_image_path(right.image_path)
    if left_path and right_path and left_path == right_path:
        return True

    if left.page is not None and right.page is not None and left.page != right.page:
        return False

    if left.bbox is not None and right.bbox is not None:
        iou = _bbox_iou(left.bbox, right.bbox)
        min_overlap = _bbox_overlap_ratio_min(left.bbox, right.bbox)
        area_ratio = _bbox_area_ratio(left.bbox, right.bbox)
        if iou >= 0.68:
            return True
        if min_overlap >= 0.92 and 0.55 <= area_ratio <= 1.82:
            return True
        if _bbox_centers_close(left.bbox, right.bbox) and 0.72 <= area_ratio <= 1.38:
            return True

    left_hash = _image_file_hash(left.image_path, image_hash_cache)
    right_hash = _image_file_hash(right.image_path, image_hash_cache)
    if left_hash and right_hash and left_hash == right_hash:
        return left.bbox is None or right.bbox is None or _bbox_iou(left.bbox, right.bbox) >= 0.25

    return False


def _merge_duplicate_figures(left: DocumentBlock, right: DocumentBlock) -> DocumentBlock:
    primary, secondary = (left, right)
    if _figure_quality_score(right) > _figure_quality_score(left):
        primary, secondary = right, left

    if not primary.image_path and secondary.image_path:
        primary.image_path = secondary.image_path
    if not primary.caption and secondary.caption:
        primary.caption = secondary.caption
    if not primary.text and secondary.text:
        primary.text = secondary.text
    if primary.bbox is None and secondary.bbox is not None:
        primary.bbox = secondary.bbox
    if primary.page is None and secondary.page is not None:
        primary.page = secondary.page
    primary.confidence = max(float(primary.confidence or 0.0), float(secondary.confidence or 0.0))
    primary.metadata = _merge_figure_metadata(primary.metadata, secondary.metadata)
    if primary.caption and not primary.text:
        primary.text = primary.caption
    return primary


def _merge_figure_metadata(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(secondary or {})
    merged.update(primary or {})
    for key, value in (secondary or {}).items():
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    if (primary or {}).get("contains_schema") or (secondary or {}).get("contains_schema"):
        merged["contains_schema"] = True
    merged["deduplicated_visual"] = True

    caption_display_values = [
        value
        for value in ((primary or {}).get("caption_display"), (secondary or {}).get("caption_display"))
        if value is not None
    ]
    if caption_display_values:
        merged["caption_display"] = any(value is not False for value in caption_display_values)

    assets: list[dict[str, Any]] = []
    seen_assets: set[tuple[str, str]] = set()
    for metadata in (secondary or {}, primary or {}):
        for asset in metadata.get("llm_assets") or []:
            if not isinstance(asset, dict) or not asset.get("path"):
                continue
            key = (str(asset.get("type") or ""), str(asset.get("path") or ""))
            if key in seen_assets:
                continue
            seen_assets.add(key)
            assets.append(dict(asset))
    if assets:
        merged["llm_assets"] = assets
    return merged


def _figure_quality_score(block: DocumentBlock) -> float:
    metadata = block.metadata or {}
    score = float(block.confidence or 0.0)
    if block.image_path:
        score += 8.0
    if block.caption or block.text:
        score += 2.0
    if metadata.get("contains_schema"):
        score += 1.5
    if metadata.get("source") == "vector_graphic_drawing":
        score += 1.4
    if metadata.get("source") == "pdf_native_image":
        score += 1.0
    if block.bbox is not None:
        score += min(4.0, (block.bbox.width * block.bbox.height) / 80_000.0)
    return score


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _bbox_intersection_area(left, right)
    if intersection <= 0:
        return 0.0
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_overlap_ratio_min(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _bbox_intersection_area(left, right)
    min_area = min(left.width * left.height, right.width * right.height)
    return intersection / min_area if min_area > 0 else 0.0


def _bbox_intersection_area(left: BoundingBox, right: BoundingBox) -> float:
    ix0, iy0 = max(left.x0, right.x0), max(left.y0, right.y0)
    ix1, iy1 = min(left.x1, right.x1), min(left.y1, right.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_area_ratio(left: BoundingBox, right: BoundingBox) -> float:
    left_area = max(1.0, left.width * left.height)
    right_area = max(1.0, right.width * right.height)
    return min(left_area, right_area) / max(left_area, right_area)


def _bbox_centers_close(left: BoundingBox, right: BoundingBox) -> bool:
    center_delta = abs(left.center_x - right.center_x) + abs(left.center_y - right.center_y)
    width_delta = abs(left.width - right.width)
    height_delta = abs(left.height - right.height)
    return center_delta <= 18.0 and width_delta <= max(12.0, min(left.width, right.width) * 0.12) and height_delta <= max(
        12.0,
        min(left.height, right.height) * 0.12,
    )


def _normalized_image_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    try:
        return str(Path(image_path).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(image_path)


def _image_file_hash(image_path: str | None, cache: dict[str, str | None]) -> str | None:
    normalized = _normalized_image_path(image_path)
    if not normalized:
        return None
    if normalized in cache:
        return cache[normalized]
    path = Path(normalized)
    if not path.exists() or not path.is_file():
        cache[normalized] = None
        return None
    try:
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    cache[normalized] = digest
    return digest


def _bbox_signature(bbox: BoundingBox) -> tuple[int, int, int, int]:
    return (
        int(round(bbox.x0)),
        int(round(bbox.y0)),
        int(round(bbox.x1)),
        int(round(bbox.y1)),
    )


def _document_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "document"
    return f"{stem}_{digest}"


def _position_key(block: DocumentBlock) -> tuple[int, float, float]:
    bbox = block.bbox
    return (int(block.page or 0), bbox.y0 if bbox else float("inf"), bbox.x0 if bbox else float("inf"))
