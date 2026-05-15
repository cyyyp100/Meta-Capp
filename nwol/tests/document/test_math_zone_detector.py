from document.layout.block_classifier import classify_blocks
from document.layout.math_zone_detector import raw_lines_to_math_aware_blocks
from document.models import BoundingBox, RawLine


def line(text, y0, y1, x0=100, x1=180, font_name=None):
    return RawLine(
        text=text,
        page=1,
        bbox=BoundingBox(x0, y0, x1, y1),
        font_size=10,
        font_name=font_name,
    )


def test_inline_formula_sentence_is_not_display_formula():
    lines = [
        line("Comme n² − n ∼ n² et ln(1 + 1/n) ∼ 1/n, on obtient", 100, 112, 70, 520),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (600, 800)})
    blocks = classify_blocks(raw_blocks)

    assert all(block.type != "formula" for block in blocks)


def test_short_prose_formula_tail_is_not_display_formula():
    lines = [
        line("naturel n, u_{n+1} = u_n + r.", 83, 97, 31, 153),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (600, 800)})
    blocks = classify_blocks(raw_blocks)

    assert len(blocks) == 1
    assert blocks[0].type == "paragraph"
    assert blocks[0].metadata["formula_mode"] == "inline"


def test_display_multiline_formula_gets_display_mode():
    lines = [
        line("u_n", 100, 110),
        line("v_n", 114, 124),
        line("→ 1", 128, 138),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (600, 800)})
    blocks = classify_blocks(raw_blocks)

    assert len(blocks) == 1
    assert blocks[0].type == "formula"
    assert blocks[0].metadata["formula_mode"] == "display"
    assert blocks[0].metadata["render_mode"] == "pdf_crop"


def test_centered_math_font_equation_with_number_is_display_formula():
    lines = [
        line("Balance = Exports −Imports.", 100, 122, 222, 373, font_name="LMMathItalic10-Regular"),
        line("(2.2)", 100, 114, 510, 533),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (595, 800)})
    blocks = classify_blocks(raw_blocks)

    assert len(blocks) == 1
    assert blocks[0].type == "formula"
    assert "(2.2)" in blocks[0].text


def test_stacked_summation_parts_are_one_display_formula():
    lines = [
        line("n", 100, 108, 305, 310, font_name="LMMathItalic8-Regular"),
        line("X", 101, 138, 300, 315, font_name="LMMathExtension10-Regular"),
        line("GDP_{EU} =", 106, 123, 247, 297, font_name="LMRoman10-Regular"),
        line("(2.1)", 106, 121, 510, 533),
        line("GDP_{i},", 110, 122, 317, 348, font_name="LMMathItalic10-Regular"),
        line("i=1", 122, 134, 301, 315, font_name="LMRoman8-Regular"),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (595, 800)})
    blocks = classify_blocks(raw_blocks)

    assert len(blocks) == 1
    assert blocks[0].type == "formula"
    assert "GDP_{EU}" in blocks[0].text
    assert "(2.1)" in blocks[0].text


def test_ambiguous_fragment_preserves_text_flow():
    lines = [
        line("1", 100, 110),
        line("n", 114, 124),
        line(")", 128, 138),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (600, 800)})

    assert any(block.text.strip() == ")" for block in raw_blocks)
    assert all(block.text.strip() for block in raw_blocks)
