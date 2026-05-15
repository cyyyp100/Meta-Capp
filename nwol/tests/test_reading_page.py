import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.reading_page import (
    _block_allows_interaction,
    _block_page_end_number,
    _build_section_text,
    _collect_section_image_paths,
)


def test_front_matter_email_block_is_not_interactive():
    block = {
        "type": "paragraph",
        "text": "Tianang Leng* Huazhong University of Science and Technology Wuhan, China tianangl@hust.edu.cn",
        "page_number": 1,
        "metadata": {},
    }

    assert _block_allows_interaction(block) is False
    assert block["is_metadata"] is True
    assert block["metadata"]["is_metadata"] is True


def test_real_content_paragraph_remains_interactive_without_displayable_flag():
    block = {
        "type": "paragraph",
        "text": (
            "The proposed module adapts the segmentation model to new classes by "
            "combining support examples with query images during inference."
        ),
        "page_number": 3,
        "metadata": {},
    }

    assert _block_allows_interaction(block) is True


def test_block_page_end_number_preserves_cross_page_context():
    block = {"page_start": 3, "page_end": 4}

    assert _block_page_end_number(block, fallback=3) == 4


def test_section_text_hides_corrupt_formula_latex():
    text = _build_section_text(
        [
            {"type": "paragraph", "text": "Fig. 1 depicts the architecture."},
            {"type": "formula", "text": r"$\ l a_{bel}{loss L_{o_i} = L_{bce} + L_{iou}}$"},
        ]
    )

    assert "[Formule affichée]" in text
    assert r"\ l a" not in text


def test_section_image_paths_include_table_crop(tmp_path):
    table_crop = tmp_path / "table.png"
    table_crop.write_bytes(b"png")

    paths = _collect_section_image_paths(
        [
            {
                "type": "table",
                "metadata": {"table_image_path": str(table_crop)},
            }
        ]
    )

    assert paths == [str(table_crop)]
