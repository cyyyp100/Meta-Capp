#!/usr/bin/env python3
"""Diagnostic script — test PDF pipeline page by page.

Usage:
    /opt/miniconda3/envs/nwol/bin/python scripts/test_pdf_pipeline.py
    /opt/miniconda3/envs/nwol/bin/python scripts/test_pdf_pipeline.py doc_test/SAM.pdf
    /opt/miniconda3/envs/nwol/bin/python scripts/test_pdf_pipeline.py doc_test/SAM.pdf --no-llm
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
import textwrap
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NWOL = _PROJECT_ROOT / "nwol"
if str(_NWOL) not in sys.path:
    sys.path.insert(0, str(_NWOL))

_DEBUG_OUT = _PROJECT_ROOT / "debug_out"

_TYPE_COLORS = {
    "heading": "#1565C0",
    "subheading": "#1976D2",
    "subsubheading": "#1E88E5",
    "paragraph": "#2E7D32",
    "formula": "#E65100",
    "figure": "#6A1B9A",
    "table": "#00695C",
    "bullet_list": "#558B2F",
    "definition": "#AD1457",
    "theorem": "#6D4C41",
    "remark": "#546E7A",
    "example": "#37474F",
    "caption": "#78909C",
    "code": "#455A64",
}
_DEFAULT_COLOR = "#9E9E9E"


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF pipeline diagnostic")
    parser.add_argument("pdf", nargs="?", help="Path to PDF (default: all in doc_test/)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM enrichment (faster)")
    args = parser.parse_args()

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = sorted((_PROJECT_ROOT / "doc_test").glob("*.pdf"))

    if not pdfs:
        print("No PDF found. Pass a path or put PDFs in doc_test/.")
        sys.exit(1)

    _DEBUG_OUT.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdfs:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")
        _process_pdf(pdf_path, no_llm=args.no_llm)


def _process_pdf(pdf_path: Path, *, no_llm: bool) -> None:
    from document.pdf_router import extract_document

    if no_llm:
        os.environ.setdefault("NWOL_DISABLE_LLM_PDF", "1")

    try:
        result = extract_document(str(pdf_path))
    except Exception as exc:
        print(f"  ERROR: extraction failed: {exc}")
        return

    blocks = result.blocks
    print(f"  Engine: {result.engine_name}  Score: {result.score:.2f}  Blocks: {len(blocks)}  Pages: {result.pages}")
    if result.warnings:
        for w in result.warnings[:5]:
            print(f"  WARN: {w}")

    by_page: dict[int, list] = {}
    for b in blocks:
        page = int(b.page or 0)
        by_page.setdefault(page, []).append(b)

    for page_num in sorted(by_page):
        page_blocks = by_page[page_num]
        print(f"\n  --- Page {page_num} ({len(page_blocks)} blocs) ---")
        for b in page_blocks:
            meta = b.metadata or {}
            render_mode = meta.get("render_mode", "")
            ctx_reason = meta.get("context_asset_reason", "")
            ctx_display = meta.get("context_asset_display", "")
            img = "+" if (b.image_path or meta.get("formula_image_path") or meta.get("context_asset_path")) else "-"
            conf = f"{float(b.confidence or 1.0):.2f}"
            text_preview = textwrap.shorten((b.text or b.latex or b.caption or ""), width=72, placeholder="…")
            flags = " ".join(filter(None, [
                render_mode,
                f"ctx:{ctx_reason}" if ctx_reason else "",
                f"ctx_display" if ctx_display else "",
            ]))
            print(f"    [{b.type:<18}] img={img} conf={conf} p={b.page} {flags}")
            if text_preview:
                print(f"      {text_preview}")

    report_path = _DEBUG_OUT / f"{pdf_path.stem}_report.html"
    _write_html_report(pdf_path, by_page, result, report_path)
    print(f"\n  Report: {report_path}")


def _write_html_report(
    pdf_path: Path,
    by_page: dict[int, list],
    result,
    report_path: Path,
) -> None:
    try:
        import fitz
        from PIL import Image, ImageDraw
    except ImportError as exc:
        print(f"  HTML report skipped (missing dependency: {exc})")
        return

    sections: list[str] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  Cannot open PDF for rendering: {exc}")
        return

    with doc:
        for page_num in sorted(by_page):
            page_blocks = by_page[page_num]
            page_index = page_num - 1
            if page_index < 0 or page_index >= len(doc):
                continue

            page = doc[page_index]
            zoom = 144 / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img, "RGBA")

            for b in page_blocks:
                if b.bbox is None:
                    continue
                color_hex = _TYPE_COLORS.get(b.type, _DEFAULT_COLOR)
                r, g, bl = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                x0 = int(b.bbox.x0 * zoom)
                y0 = int(b.bbox.y0 * zoom)
                x1 = int(b.bbox.x1 * zoom)
                y1 = int(b.bbox.y1 * zoom)
                draw.rectangle([x0, y0, x1, y1], outline=(r, g, bl, 220), width=2)
                draw.rectangle([x0, y0, x1, min(y0 + 14, y1)], fill=(r, g, bl, 80))

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            block_rows: list[str] = []
            for b in page_blocks:
                meta = b.metadata or {}
                render_mode = meta.get("render_mode", "")
                ctx_reason = meta.get("context_asset_reason", "")
                ctx_display = "✓" if meta.get("context_asset_display") else ""
                img_flag = "✓" if (b.image_path or meta.get("formula_image_path") or meta.get("context_asset_path")) else ""
                conf = f"{float(b.confidence or 1.0):.2f}"
                text_preview = html.escape(textwrap.shorten(b.text or b.latex or b.caption or "", width=90, placeholder="…"))
                color = _TYPE_COLORS.get(b.type, _DEFAULT_COLOR)
                block_rows.append(
                    f'<tr>'
                    f'<td style="color:{color};font-weight:bold">{html.escape(b.type)}</td>'
                    f'<td>{conf}</td>'
                    f'<td>{html.escape(render_mode)}</td>'
                    f'<td>{html.escape(ctx_reason)}</td>'
                    f'<td style="text-align:center">{ctx_display}</td>'
                    f'<td style="text-align:center">{img_flag}</td>'
                    f'<td style="max-width:400px;word-break:break-word">{text_preview}</td>'
                    f'</tr>'
                )

            section = f"""
<div class="page-section">
  <h2>Page {page_num} — {len(page_blocks)} blocs</h2>
  <div class="page-layout">
    <div class="page-render">
      <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border:1px solid #ccc">
    </div>
    <div class="page-blocks">
      <table>
        <thead><tr>
          <th>Type</th><th>Conf</th><th>render_mode</th>
          <th>ctx_reason</th><th>ctx_disp</th><th>img</th><th>Text</th>
        </tr></thead>
        <tbody>{''.join(block_rows)}</tbody>
      </table>
    </div>
  </div>
</div>"""
            sections.append(section)

    legend_rows = "".join(
        f'<span style="background:{c};color:#fff;padding:2px 8px;margin:2px;border-radius:3px;font-size:12px">{html.escape(t)}</span>'
        for t, c in _TYPE_COLORS.items()
    )

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>PDF Pipeline Report — {html.escape(pdf_path.name)}</title>
<style>
  body {{ font-family: monospace; font-size: 13px; margin: 20px; background: #fafafa; }}
  h1 {{ font-size: 18px; border-bottom: 2px solid #333; padding-bottom: 6px; }}
  h2 {{ font-size: 14px; margin: 24px 0 8px; color: #333; }}
  .meta {{ background: #e8f0fe; padding: 10px; border-radius: 4px; margin-bottom: 16px; }}
  .legend {{ margin-bottom: 16px; }}
  .page-section {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin-bottom: 24px; background: #fff; }}
  .page-layout {{ display: flex; gap: 16px; align-items: flex-start; }}
  .page-render {{ flex: 0 0 auto; max-width: 45%; }}
  .page-blocks {{ flex: 1; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #f0f0f0; border: 1px solid #ccc; padding: 4px 8px; text-align: left; font-size: 11px; }}
  td {{ border: 1px solid #eee; padding: 3px 8px; vertical-align: top; font-size: 11px; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>PDF Pipeline Report — {html.escape(pdf_path.name)}</h1>
<div class="meta">
  Engine: <b>{html.escape(result.engine_name)}</b> &nbsp;|&nbsp;
  Score: <b>{result.score:.3f}</b> &nbsp;|&nbsp;
  Blocks: <b>{len(result.blocks)}</b> &nbsp;|&nbsp;
  Pages: <b>{result.pages}</b>
</div>
<div class="legend"><b>Types :</b> {legend_rows}</div>
{''.join(sections)}
</body>
</html>"""

    report_path.write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    main()
