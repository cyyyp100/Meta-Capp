import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from document.models import BoundingBox, DocumentBlock
from document.postprocess.vector_graphics import (
    _exclude_nearby_non_graphic_text,
    _is_semantic_diagram_text_block,
    _is_text_inside_graphic,
    _strip_embedded_diagram_suffix,
)
from document.postprocess.vector_graphics import _is_graphic_label_candidate


def test_small_formula_legend_can_be_absorbed_into_graph_crop():
    block = DocumentBlock(
        type="formula",
        text="MAML + + seed_0",
        page=2,
        bbox=BoundingBox(410.0, 132.0, 460.0, 150.0),
        image_path="/tmp/formula.png",
        metadata={"formula_mode": "display"},
    )

    assert _is_graphic_label_candidate(block) is True


def test_caption_is_not_graphic_label_candidate():
    block = DocumentBlock(
        type="paragraph",
        text="Figure 1: Stabilizing MAML: this figure illustrates three seeds.",
        page=2,
        bbox=BoundingBox(108.0, 259.0, 504.0, 269.0),
        metadata={"is_caption": True},
    )

    assert _is_graphic_label_candidate(block) is False


def test_multi_word_axis_label_is_graphic_label_candidate():
    block = DocumentBlock(
        type="paragraph",
        text="Validation accuracy",
        page=2,
        bbox=BoundingBox(42.0, 118.0, 55.0, 205.0),
    )

    assert _is_graphic_label_candidate(block) is True


def test_short_heading_inside_graphic_is_treated_as_label():
    label = DocumentBlock(
        type="heading",
        text="partnership",
        page=1,
        bbox=BoundingBox(394, 532, 444, 547),
    )

    class Rect:
        x0 = 100
        y0 = 350
        x1 = 500
        y1 = 560

    assert _is_text_inside_graphic(label, Rect()) is True

    merged_labels = DocumentBlock(
        type="paragraph",
        text="European Union Russia security tension",
        page=1,
        bbox=BoundingBox(142, 456, 333, 547),
    )

    assert _is_text_inside_graphic(merged_labels, Rect()) is True

    prose = DocumentBlock(
        type="paragraph",
        text="critical technologies and defend essential interests without excessive dependence on another power.",
        page=1,
        bbox=BoundingBox(62, 352, 533, 367),
    )

    assert _is_text_inside_graphic(prose, Rect()) is False

    real_heading = DocumentBlock(
        type="heading",
        text="3.2 Power as a Combination",
        page=1,
        bbox=BoundingBox(62, 590, 272, 610),
    )

    assert _is_text_inside_graphic(real_heading, Rect()) is False


def test_repeated_diagram_terms_inside_graphic_are_treated_as_labels():
    class Rect:
        x0 = 199
        y0 = 60
        x1 = 462
        y1 = 191
        width = 263

    label = DocumentBlock(
        type="paragraph",
        text="Transformer layer Transformer layer Transformer layer Flexible Mask Adapter Adapter",
        page=3,
        bbox=BoundingBox(297, 122, 438, 166),
    )

    assert _is_text_inside_graphic(label, Rect()) is True

    short_label = DocumentBlock(
        type="paragraph",
        text="Self-Sampling",
        page=3,
        bbox=BoundingBox(253, 92, 290, 98),
    )

    assert _is_graphic_label_candidate(short_label) is True


def test_vector_crop_excludes_prose_line_above_graph():
    import fitz

    rect = fitz.Rect(80, 340, 500, 560)
    drawing_rect = fitz.Rect(108, 351, 487, 553)
    prose = DocumentBlock(
        type="paragraph",
        text="critical technologies and defend essential interests without excessive dependence on another power.",
        page=1,
        bbox=BoundingBox(62, 352, 533, 367),
    )
    label = DocumentBlock(
        type="paragraph",
        text="China",
        page=1,
        bbox=BoundingBox(406, 379, 433, 393),
    )

    adjusted = _exclude_nearby_non_graphic_text(rect, drawing_rect, [prose, label], [label])

    assert adjusted.y0 >= prose.bbox.y1 + 4


def test_vector_crop_does_not_cut_compact_diagram_when_prose_bbox_overlaps_top():
    import fitz

    rect = fitz.Rect(115, 495, 480, 628)
    drawing_rect = fitz.Rect(127, 507, 468, 616)
    prose = DocumentBlock(
        type="paragraph",
        text=(
            "This formula is not a real measurement; it is a pedagogical "
            "representation showing that influence is multidimensional. Ancient Greece"
        ),
        page=4,
        bbox=BoundingBox(62, 406, 533, 521),
    )

    adjusted = _exclude_nearby_non_graphic_text(rect, drawing_rect, [prose], [])

    assert adjusted.y0 == rect.y0


def test_embedded_diagram_suffix_is_removed_from_prose():
    text = (
        "This formula is not a real measurement; it is a pedagogical "
        "representation showing that influence is multidimensional. Ancient Greece"
    )

    assert _strip_embedded_diagram_suffix(text).endswith("multidimensional.")


def test_embedded_climate_graph_labels_are_removed_from_prose():
    text = (
        "Europe can reduce emissions by improving energy efficiency, changing "
        "production methods and increasing low-carbon energy. Emissions Desired "
        "decarbonization path"
    )

    assert _strip_embedded_diagram_suffix(text).endswith("low-carbon energy.")


def test_embedded_soft_power_diagram_suffix_is_removed_from_prose():
    text = (
        "A researcher who collaborates with European laboratories participates in this "
        "soft power. Values European Culture Education Soft Power Diplomacy"
    )

    assert _strip_embedded_diagram_suffix(text).endswith("soft power.")


def test_semantic_diagram_text_block_is_absorbed_by_vector_crop():
    class Rect:
        x0 = 127
        y0 = 507
        x1 = 468
        y1 = 616
        width = 341
        height = 109

    block = DocumentBlock(
        type="paragraph",
        text=(
            "Ancient Greece philosophy Renaissance arts Modern Europe science, "
            "industry, politics Figure 1.1: A simplified historical chain of "
            "European influence. Roman Empire law"
        ),
        page=4,
        bbox=BoundingBox(133, 507, 455, 637),
        metadata={"semantic_only_block": True, "source": "opendataloader_pdf"},
    )

    assert _is_semantic_diagram_text_block(block, Rect()) is True
