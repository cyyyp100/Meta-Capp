# tests/test_segmentation.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.segmentation import segment_blocks


def test_formula_display_detection():
    blocks = [{"type": "paragraph", "text": "$$\\int_a^b f(x)dx$$"}]
    result = segment_blocks(blocks)
    assert result[0]["type"] == "formula"
    assert result[0]["display"] is True
    assert "\\int" in result[0]["latex"]


def test_formula_inline_detection():
    blocks = [{"type": "paragraph", "text": "$E = mc^2$"}]
    result = segment_blocks(blocks)
    assert result[0]["type"] == "formula"
    assert result[0]["display"] is False


def test_code_detection():
    blocks = [{"type": "paragraph", "text": "def f(x):\n    return x**2"}]
    result = segment_blocks(blocks)
    assert result[0]["type"] == "code"


def test_for_phrase_is_not_code():
    blocks = [{"type": "paragraph", "text": "for testing."}]
    result = segment_blocks(blocks)
    assert result[0]["type"] == "paragraph"


def test_for_loop_is_code():
    blocks = [{"type": "paragraph", "text": "for item in items:"}]
    result = segment_blocks(blocks)
    assert result[0]["type"] == "code"


def test_displayable_false_code_fragment_is_filtered():
    blocks = [{"type": "paragraph", "text": "for testing.", "metadata": {"displayable": False}}]
    result = segment_blocks(blocks)
    assert result == []


def test_displayable_false_bullet_list_is_filtered():
    blocks = [{"type": "bullet_list", "items": ["Table 1 duplicated text"], "metadata": {"displayable": False}}]
    result = segment_blocks(blocks)
    assert result == []


def test_caption_detection():
    blocks = [{"type": "paragraph", "text": "Figure 3 : Réseau de neurones"}]
    result = segment_blocks(blocks)
    assert result == []


def test_caption_is_attached_to_previous_figure_before_filtering():
    blocks = [
        {
            "type": "figure",
            "page_number": 6,
            "bbox": [108, 371, 487, 553],
            "image_path": "/tmp/schema.png",
            "caption": "",
            "metadata": {"contains_schema": True, "caption_display": False},
        },
        {
            "type": "paragraph",
            "text": "Figure 3.1: Europe between cooperation, competition and security challenges.",
            "page_number": 6,
            "bbox": [128, 557, 468, 571],
            "is_caption": True,
            "metadata": {"is_caption": True, "displayable": True},
        },
    ]

    result = segment_blocks(blocks)

    assert [block["type"] for block in result] == ["figure"]
    assert result[0]["caption"] == "Figure 3.1: Europe between cooperation, competition and security challenges."
    assert result[0]["metadata"]["caption_display"] is True


def test_unattached_caption_with_visual_assets_remains_visible():
    blocks = [
        {
            "type": "paragraph",
            "text": "Figure 2: Important architecture overview.",
            "page_number": 3,
            "is_caption": True,
            "metadata": {
                "is_caption": True,
                "displayable": True,
                "visual_assets": [{"asset_type": "schema", "image_path": "/tmp/schema.png"}],
            },
        },
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["Figure 2: Important architecture overview."]


def test_heading_preserved():
    blocks = [{"type": "heading", "level": 1, "text": "Chapitre 1"}]
    result = segment_blocks(blocks)
    assert result[0]["type"] == "heading"


def test_large_short_course_heading_is_preserved():
    blocks = [{
        "type": "heading",
        "level": 1,
        "text": "Environmental Leadership",
        "bbox": [62, 169, 383, 205],
        "metadata": {"font_size": 24.7, "page_height": 841.0},
    }]

    result = segment_blocks(blocks)

    assert result[0]["text"] == "Environmental Leadership"


def test_additional_results_heading_is_preserved():
    blocks = [{
        "type": "heading",
        "text": "ADDITIONAL RESULTS",
        "page_number": 10,
        "bbox": [129, 522, 245, 537],
        "metadata": {"font_size": 11.9, "page_height": 792.0},
    }]

    result = segment_blocks(blocks)

    assert result[0]["text"] == "ADDITIONAL RESULTS"


def test_semantic_heading_with_body_is_dropped_when_native_heading_follows():
    blocks = [
        {
            "type": "heading",
            "text": "3.2 Power as a Combination Geopolitical power can be represented with a simple model:",
            "page_number": 6,
            "metadata": {"semantic_only_block": True},
        },
        {"type": "heading", "text": "3.2 Power as a Combination", "page_number": 6},
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["3.2 Power as a Combination"]


def test_table_of_contents_page_is_hidden_from_reader_flow():
    blocks = [
        {"type": "heading", "text": "Contents", "page_number": 3},
        {"type": "bullet_list", "items": ["1 Chapter 1"], "page_number": 3, "metadata": {"displayable": True}},
        {"type": "heading", "text": "Chapter 1", "page_number": 4},
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["Chapter 1"]


def test_repeated_multiline_heading_tail_is_dropped():
    blocks = [
        {
            "type": "heading",
            "text": "The European Union as an Economic Actor",
            "page_number": 5,
            "bbox": [62, 169, 518, 205],
            "metadata": {"font_size": 24.7, "page_height": 841.0},
        },
        {
            "type": "heading",
            "text": "Actor",
            "page_number": 5,
            "bbox": [62, 199, 131, 234],
            "metadata": {"font_size": 24.7, "page_height": 841.0},
        },
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["The European Union as an Economic Actor"]


def test_numbered_semantic_heading_with_optional_dot_is_dropped():
    blocks = [
        {
            "type": "heading",
            "text": "5.1. A Simple Climate Equation Carbon emissions can be represented in a simplified way:",
            "page_number": 8,
            "metadata": {"semantic_only_block": True},
        },
        {"type": "heading", "text": "5.1 A Simple Climate Equation", "page_number": 8},
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["5.1 A Simple Climate Equation"]


def test_formula_fragment_inside_previous_formula_is_dropped():
    blocks = [
        {
            "type": "formula",
            "text": "GDP_{EU} = sum GDP_i",
            "page_number": 5,
            "bbox": [247, 407, 533, 445],
            "image_path": "/tmp/full.png",
        },
        {
            "type": "formula",
            "text": "$i = 1$",
            "page_number": 5,
            "bbox": [301, 429, 315, 440],
            "image_path": "/tmp/fragment.png",
            "metadata": {"semantic_only_block": True},
        },
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["GDP_{EU} = sum GDP_i"]


def test_overlapping_semantic_formula_duplicate_is_dropped():
    blocks = [
        {
            "type": "formula",
            "text": "duplicate semantic crop",
            "page_number": 3,
            "bbox": [243, 667, 504, 735],
            "image_path": "/tmp/semantic.png",
            "metadata": {"semantic_only_block": True},
        },
        {
            "type": "formula",
            "text": "clean native crop",
            "page_number": 3,
            "bbox": [244, 704, 504, 742],
            "image_path": "/tmp/native.png",
        },
    ]

    result = segment_blocks(blocks)

    assert [block["text"] for block in result] == ["clean native crop"]


def test_short_paragraph_tail_is_merged_before_visibility_filter():
    blocks = [
        {
            "type": "paragraph",
            "text": "Europe has strong soft power because of its universities, museums, political values, cultural industries and social",
            "page_number": 9,
            "bbox": [62, 338, 533, 367],
            "metadata": {"displayable": True},
        },
        {
            "type": "paragraph",
            "text": "models.",
            "page_number": 9,
            "bbox": [62, 365, 98, 381],
            "metadata": {"displayable": False},
        },
    ]

    result = segment_blocks(blocks)

    assert result[0]["text"].endswith("social models.")


def test_duplicate_acronym_formula_after_text_is_dropped():
    blocks = [
        {"type": "abstract", "text": "We call the method MAML++.", "page_number": 1, "metadata": {"displayable": True}},
        {
            "type": "formula",
            "text": "$MAML + +.$",
            "page_number": 1,
            "bbox": [143, 372, 188, 384],
            "image_path": "/tmp/maml.png",
        },
    ]

    result = segment_blocks(blocks)

    assert [block["type"] for block in result] == ["abstract"]


def test_acronym_formula_after_introductory_text_is_merged():
    blocks = [
        {"type": "abstract", "text": "We call the method", "page_number": 1, "metadata": {"displayable": True}},
        {
            "type": "formula",
            "text": "$MAML + +.$",
            "page_number": 1,
            "bbox": [143, 372, 188, 384],
            "image_path": "/tmp/maml.png",
        },
    ]

    result = segment_blocks(blocks)

    assert [block["type"] for block in result] == ["abstract"]
    assert result[0]["text"] == "We call the method MAML++."


def test_visual_legend_overlapping_figure_is_dropped():
    blocks = [
        {"type": "figure", "page_number": 2, "bbox": [150, 70, 488, 245], "image_path": "/tmp/fig.png"},
        {
            "type": "bullet_list",
            "items": ["MAML++ seed_0", "MAML++ seed_1"],
            "page_number": 2,
            "bbox": [410, 132, 460, 159],
            "metadata": {"displayable": True},
        },
    ]

    result = segment_blocks(blocks)

    assert [block["type"] for block in result] == ["figure"]


def test_orphan_lowercase_bullet_is_converted_to_paragraph():
    blocks = [{
        "type": "bullet_list",
        "items": ["• that learns how to update a base-learner model."],
        "page_number": 3,
        "metadata": {"displayable": True},
    }]

    result = segment_blocks(blocks)

    assert result[0]["type"] == "paragraph"
    assert result[0]["text"].startswith("that learns")


def test_orphan_denotes_bullet_is_dropped():
    blocks = [{
        "type": "bullet_list",
        "items": ["• denotes the target set loss of task b"],
        "page_number": 5,
        "metadata": {"displayable": True},
    }]

    assert segment_blocks(blocks) == []


def test_empty_table_overlapping_figure_is_dropped():
    blocks = [
        {"type": "figure", "page_number": 2, "bbox": [150, 70, 488, 245], "image_path": "/tmp/fig.png"},
        {
            "type": "table",
            "text": "",
            "page_number": 2,
            "bbox": [160, 82, 478, 235],
            "image_path": "/tmp/empty-table.png",
        },
    ]

    result = segment_blocks(blocks)

    assert [block["type"] for block in result] == ["figure"]


def test_corrupt_inline_dollar_spacing_before_denotes_is_repaired():
    blocks = [{
        "type": "paragraph",
        "text": "The term $v_{i}denotes the validation split.",
        "page_number": 5,
        "metadata": {"displayable": True},
    }]

    result = segment_blocks(blocks)

    assert result[0]["text"] == "The term v_{i} denotes the validation split."


def test_corrupt_inline_dollar_around_arrow_word_is_repaired():
    blocks = [{
        "type": "paragraph",
        "text": r"Gradient Instabilit$y \rightarrow M$ulti-Step Loss Optimization.",
        "page_number": 5,
        "metadata": {"displayable": True},
    }]

    result = segment_blocks(blocks)

    assert result[0]["text"] == r"Gradient Instability \rightarrow Multi-Step Loss Optimization."


def test_reference_line_misclassified_as_heading_is_dropped():
    blocks = [
        {
            "type": "heading",
            "text": "Methodology and computing in applied probability, 1(2):127-190, 1999.",
            "page_number": 10,
        }
    ]

    assert segment_blocks(blocks) == []


# tests/test_scopes.py
from core.scopes import make_chapter_scope, make_page_scope, TextScope


def test_chapter_scope_plain_text():
    blocks = [
        {"type": "paragraph", "text": "Bonjour le monde"},
        {"type": "formula", "latex": "x^2"},
        {"type": "paragraph", "text": "Suite du cours"},
    ]
    scope = make_chapter_scope("Chap 1", 1, 3, blocks)
    text = scope.plain_text()
    assert "Bonjour le monde" in text
    assert "x^2" in text
    assert "Suite du cours" in text


def test_page_scope():
    scope = make_page_scope(5, [{"type": "paragraph", "text": "Test"}])
    assert scope.scope_type == "page"
    assert scope.page_start == 5
    assert scope.label == "Page 5"


def test_scope_to_dict():
    scope = make_chapter_scope("Test", 2, 4, [])
    d = scope.to_dict()
    assert d["type"] == "chapter"
    assert d["page_start"] == 2
    assert "scope_id" in d
