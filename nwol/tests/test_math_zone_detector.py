from document.layout.math_zone_detector import raw_lines_to_math_aware_blocks
from document.models import BoundingBox, RawLine


def line(text, y0, y1, x0=100, x1=160):
    return RawLine(
        text=text,
        page=1,
        bbox=BoundingBox(x0, y0, x1, y1),
        font_size=10,
    )


def test_groups_multiline_formula():
    lines = [
        line("un", 100, 110),
        line("vn", 114, 124),
        line("→ 1", 128, 138),
    ]
    blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (600, 800)})
    assert len(blocks) == 1
    assert blocks[0].block_type == "formula_display_candidate"
    assert "un" in blocks[0].text
    assert "vn" in blocks[0].text


def test_does_not_group_normal_sentence():
    lines = [
        line("On dit que deux suites sont équivalentes.", 100, 112, 80, 380),
    ]
    blocks = raw_lines_to_math_aware_blocks(lines, page_sizes={1: (600, 800)})
    assert len(blocks) == 1
    assert blocks[0].block_type == "line"
