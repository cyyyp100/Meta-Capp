from __future__ import annotations

import re
from collections import Counter

from document.models import DocumentBlock, ExtractionResult
from document.postprocess.latex_quality import latex_looks_corrupt


def evaluate_blocks(blocks: list[DocumentBlock], pages: int = 1) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if not blocks:
        return 0.0, ["Aucun bloc lisible extrait du PDF."]

    score = 1.0
    texts = [_block_text(block) for block in blocks]
    joined = "\n".join(texts)

    unknown = joined.count("�")
    if unknown:
        score -= min(0.25, unknown * 0.02)
        warnings.append("Caractères inconnus détectés dans le texte extrait.")

    glued = _glued_word_or_symbol_count(blocks)
    if glued:
        score -= min(0.20, glued * 0.015)
        warnings.append("Des mots ou symboles semblent encore collés.")

    long_paragraphs = [
        text
        for block, text in zip(blocks, texts)
        if block.type == "paragraph" and len(text) > 1200 and not _block_has_visual_render(block)
    ]
    if long_paragraphs:
        score -= min(0.20, len(long_paragraphs) * 0.04)
        warnings.append(f"{len(long_paragraphs)} paragraphe(s) sont anormalement longs.")

    empty_blocks = [
        block
        for block, text in zip(blocks, texts)
        if not text.strip() and block.type not in {"figure"} and not _block_has_visual_render(block)
    ]
    if empty_blocks:
        score -= min(0.15, len(empty_blocks) * 0.02)
        warnings.append(f"{len(empty_blocks)} bloc(s) vide(s) détecté(s).")

    if pages >= 3 and _repeated_margin_text_remaining(blocks):
        score -= 0.10
        warnings.append("Des en-têtes ou pieds de page répétés semblent rester.")

    if _probably_mixed_columns(blocks):
        score -= 0.12
        warnings.append("L'ordre de lecture semble mélanger plusieurs colonnes.")

    risky_formulas = _math_reconstruction_risks(blocks)
    if risky_formulas:
        score -= min(0.18, risky_formulas * 0.03)
        warnings.append("Certaines formules LaTeX semblent encore mal reconstruites.")

    fragmented_inline = _fragmented_inline_formula_risk(blocks)
    if fragmented_inline:
        score -= min(0.16, fragmented_inline * 0.04)
        warnings.append("Certaines formules inline semblent avoir été fragmentées.")

    headings = sum(1 for block in blocks if block.type == "heading")
    paragraphs = sum(1 for block in blocks if block.type == "paragraph")
    if pages >= 3 and paragraphs >= 8 and headings == 0:
        score -= 0.08
        warnings.append("Très peu de titres ont été détectés.")

    return round(max(0.0, min(1.0, score)), 3), warnings


def update_result_quality(result: ExtractionResult) -> ExtractionResult:
    score, warnings = evaluate_blocks(result.blocks, pages=result.pages)
    merged_warnings = list(dict.fromkeys([*result.warnings, *warnings]))
    result.score = score
    result.warnings = merged_warnings
    return result


def _block_text(block: DocumentBlock) -> str:
    if block.type == "bullet_list":
        return " ".join(block.items or [])
    if block.type == "formula":
        return block.latex or block.text
    if block.type == "table":
        return block.text or block.markdown or block.html or ""
    return block.text or block.caption or ""


def _repeated_margin_text_remaining(blocks: list[DocumentBlock]) -> bool:
    keys: Counter[str] = Counter()
    for block in blocks:
        if block.bbox is None or block.page is None:
            continue
        if block.bbox.y0 > 90 and block.bbox.y1 < 700:
            continue
        text = re.sub(r"\d+", "<num>", _block_text(block).casefold())
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 3:
            keys[text] += 1
    return any(count >= 3 for count in keys.values())


def _probably_mixed_columns(blocks: list[DocumentBlock]) -> bool:
    last_page = None
    last_x = None
    switches = 0
    candidates = 0
    for block in blocks:
        if block.bbox is None or block.page is None or block.type not in {"paragraph", "heading"}:
            continue
        text = _block_text(block).strip()
        if len(text) < 12 or block.metadata.get("is_caption") or block.metadata.get("is_metadata"):
            continue
        # Wide blocks in a one-column article often straddle the fixed 300pt
        # threshold used below; counting them creates false mixed-column alerts.
        if block.bbox.width > 360:
            last_page = block.page
            last_x = None
            continue
        x_side = 0 if block.bbox.center_x < 300 else 1
        if block.page == last_page and last_x is not None and x_side != last_x:
            switches += 1
        candidates += 1
        last_page = block.page
        last_x = x_side
    return candidates >= 12 and switches > max(8, candidates * 0.35)


def _math_reconstruction_risks(blocks: list[DocumentBlock]) -> int:
    risks = 0
    for block in blocks:
        text = _block_text(block)
        if block.type == "formula":
            if _block_has_visual_render(block):
                continue
            if latex_looks_corrupt(text):
                risks += 1
            if text.count("{") != text.count("}"):
                risks += 1
            if re.search(r"\^\{\s*\}\s*\^|_\{\s*\}\s*_|\\[A-Za-z]+_\{\\[A-Za-z]+_\{", text):
                risks += 1
            if re.search(r"\[[^\]]*\\[A-Za-z]+[^\]]*\]", text) and not text.strip().startswith(r"\left["):
                risks += 1
        elif block.type == "paragraph":
            if _block_has_visual_render(block):
                continue
            if _paragraph_bracket_math_looks_corrupt(text):
                risks += 1
    return risks


def _block_has_visual_render(block: DocumentBlock) -> bool:
    metadata = block.metadata or {}
    return bool(
        block.image_path
        or metadata.get("formula_image_path")
        or metadata.get("table_image_path")
        or metadata.get("context_asset_path")
    )


def _paragraph_bracket_math_looks_corrupt(text: str) -> bool:
    for match in re.finditer(r"\[[^\]]*(?:\\sim|\\frac|\\rightarrow|_\{|\\int|\\sum)[^\]]*\]", text):
        segment = match.group(0)
        if latex_looks_corrupt(segment):
            return True
        if segment.count("{") != segment.count("}"):
            return True
        if len(segment) > 90 and segment.count("$") % 2 == 1:
            return True
    return False


def _glued_word_or_symbol_count(blocks: list[DocumentBlock]) -> int:
    count = 0
    for block in blocks:
        if block.type in {"formula", "table"}:
            continue
        text = _block_text(block)
        count += len(re.findall(r"[Ωωα-ω][A-Za-zÀ-ÿ]", text))
        for match in re.finditer(r"[a-zà-ÿ][A-ZÀ-Ÿ]{2,}", text):
            token = _token_around(text, match.start(), match.end())
            if _looks_like_scientific_camel_token(token):
                continue
            count += 1
    return count


def _token_around(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and re.match(r"[A-Za-zÀ-ÿ0-9'’_-]", text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and re.match(r"[A-Za-zÀ-ÿ0-9'’_-]", text[right]):
        right += 1
    return text[left:right]


def _looks_like_scientific_camel_token(token: str) -> bool:
    if not token:
        return False
    token = re.sub(r"[’']s$", "", token)
    if token in {"NIfTI"}:
        return True
    if "-" in token and re.search(r"[A-Z]{2,}", token):
        return True
    if re.search(r"\d", token):
        return True
    if re.fullmatch(r"[A-Za-z]?[a-z]{1,3}[A-Z]{2,}[A-Za-z]*", token):
        return True
    if re.fullmatch(r"[A-Z][a-z]+(?:[A-Z]{2,}|[A-Z][a-z]+)[A-Za-z]*", token):
        return True
    if re.fullmatch(r"[A-Za-z]+(?:GAN|CNN|RNN|UNet|UNetR|UNETR|SGD|MAML|AI|LU|RR|MRI|CT|DNN|SD)s?", token):
        return True
    return False


def _fragmented_inline_formula_risk(blocks: list[DocumentBlock]) -> int:
    risks = 0
    for previous, current, nxt in zip(blocks, blocks[1:], blocks[2:]):
        if current.type != "formula":
            continue
        if current.metadata.get("formula_mode") != "display":
            continue

        previous_text = (previous.text or "").strip()
        next_text = (nxt.text or "").strip()
        formula_text = (current.text or current.latex or "").strip()
        if len(formula_text) > 20:
            continue

        if previous.type == "paragraph" and previous_text.endswith(("ln", "log", "exp", "(", "+", "-", "=", "/")):
            risks += 1
        if nxt.type == "paragraph" and next_text.startswith((")", "/", "+", "-", "=", "1/")):
            risks += 1
    return risks
