from document.models import BoundingBox, DocumentBlock
from document.postprocess.math_fragments import repair_display_math_fragments
from document.postprocess.math_normalizer import normalize_math_blocks
from document.postprocess.paragraph_rebuilder import rebuild_paragraphs


def block(text, y0, y1, metadata=None):
    return DocumentBlock(
        type="paragraph",
        text=text,
        page=1,
        bbox=BoundingBox(60, y0, 500, y1),
        metadata=dict(metadata or {}),
    )


def test_inline_math_paragraphs_can_be_merged_with_surrounding_text():
    blocks = [
        block("Comme n² − n ∼ n²", 100, 112),
        block(
            "et ln(1 + 1/n) ∼ 1/n,",
            114,
            126,
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
        block("on obtient le résultat.", 128, 140),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 1
    assert result[0].type == "paragraph"
    assert "ln(1 + 1/n)" in result[0].text


def test_large_but_related_gap_can_continue_math_prose_paragraph():
    blocks = [
        block("Comme n² − n ∼ n² et", 100, 112),
        block("ln(1 + 1/n) ∼ 1/n, on obtient", 132, 144),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 1
    assert result[0].text == "Comme n² − n ∼ n² et ln(1 + 1/n) ∼ 1/n, on obtient"


def test_new_paragraph_kept_after_hard_gap():
    blocks = [
        block("On obtient le résultat.", 100, 112),
        block("Donc u_n tend vers +∞.", 180, 192),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 2
    assert result[1].text.startswith("Donc")


def test_cross_page_hyphenated_paragraph_is_merged():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text=(
                "Faster R-CNN system is known as a two-stage pipeline. The first stage is a "
                "region proposal network (RPN), receiving an image xi to produce the candidate "
                "object bounding-"
            ),
            page=3,
            bbox=BoundingBox(308, 678, 545, 714),
        ),
        DocumentBlock(
            type="paragraph",
            text=(
                "boxes (so-called object region proposals) in this image. The second stage, "
                "i.e., Fast R-CNN, shares the RPN backbone."
            ),
            page=4,
            bbox=BoundingBox(50, 73, 286, 115),
        ),
    ]

    result = rebuild_paragraphs(blocks, page_sizes={3: (612, 792), 4: (612, 792)})

    assert len(result) == 1
    assert "object bounding boxes (so-called object region proposals)" in result[0].text
    assert result[0].metadata["merged_across_pages"] is True
    assert result[0].metadata["page_end"] == 4
    assert result[0].to_reader_dict()["page_end"] == 4


def test_completed_page_paragraph_is_not_merged_with_next_page():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="This paragraph ends cleanly.",
            page=1,
            bbox=BoundingBox(60, 700, 500, 714),
        ),
        DocumentBlock(
            type="paragraph",
            text="The next page starts a new paragraph.",
            page=2,
            bbox=BoundingBox(60, 80, 500, 94),
        ),
    ]

    result = rebuild_paragraphs(blocks, page_sizes={1: (600, 800), 2: (600, 800)})

    assert [item.text for item in result] == [
        "This paragraph ends cleanly.",
        "The next page starts a new paragraph.",
    ]


def test_numbered_heading_is_not_merged_across_page_boundary():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="3.2.2 Multi-Head Attention",
            page=4,
            bbox=BoundingBox(108, 629, 231, 642),
        ),
        DocumentBlock(
            type="paragraph",
            text="output values. These are concatenated and once again projected.",
            page=5,
            bbox=BoundingBox(108, 73, 504, 86),
        ),
    ]

    result = rebuild_paragraphs(blocks, page_sizes={4: (612, 792), 5: (612, 792)})

    assert [block.text for block in result] == [
        "3.2.2 Multi-Head Attention",
        "output values. These are concatenated and once again projected.",
    ]


def test_low_next_page_text_after_visual_interlude_is_not_cross_page_merged():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="The module is illustrated in Fig. 2 and",
            page=1,
            bbox=BoundingBox(308, 700, 545, 714),
        ),
        DocumentBlock(
            type="figure",
            text="Figure 3. Diagram",
            page=2,
            bbox=BoundingBox(100, 70, 500, 290),
        ),
        DocumentBlock(
            type="paragraph",
            text="can be summarized as follows with several equations below.",
            page=2,
            bbox=BoundingBox(50, 312, 286, 324),
        ),
    ]

    result = rebuild_paragraphs(blocks, page_sizes={1: (612, 792), 2: (612, 792)})

    assert [block.text for block in result] == [
        "The module is illustrated in Fig. 2 and",
        "Figure 3. Diagram",
        "can be summarized as follows with several equations below.",
    ]


def test_orphan_math_prefix_before_connector_is_removed():
    blocks = [
        block("n Comme n² − n ∼ n² et ln(1 + 1/n) ∼ 1/n,", 100, 112),
        block("on obtient le résultat.", 114, 126),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 1
    assert result[0].text.startswith("Comme")


def test_column_restart_does_not_merge_with_previous_column_tail():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="Fin de la colonne gauche.",
            page=1,
            bbox=BoundingBox(50, 680, 285, 692),
        ),
        DocumentBlock(
            type="paragraph",
            text="Début de la colonne droite.",
            page=1,
            bbox=BoundingBox(310, 120, 545, 132),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert [block.text for block in result] == [
        "Fin de la colonne gauche.",
        "Début de la colonne droite.",
    ]


def test_parallel_column_lines_do_not_merge_when_ordered_by_y_then_x():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="Left-column sentence.",
            page=1,
            bbox=BoundingBox(50, 220, 270, 232),
        ),
        DocumentBlock(
            type="paragraph",
            text="Right-column sentence.",
            page=1,
            bbox=BoundingBox(310, 221, 545, 233),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert [block.text for block in result] == [
        "Left-column sentence.",
        "Right-column sentence.",
    ]


def test_adjacent_cross_column_lines_do_not_merge_after_hyphenated_tail():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="queries to visual prompts that lie in the same visual embed-",
            page=1,
            bbox=BoundingBox(309, 678, 545, 690),
        ),
        DocumentBlock(
            type="paragraph",
            text="the cosine annealing learning rate as recommended in [1] to",
            page=1,
            bbox=BoundingBox(50, 690, 286, 702),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert [block.text for block in result] == [
        "queries to visual prompts that lie in the same visual embed-",
        "the cosine annealing learning rate as recommended in [1] to",
    ]


def test_soft_hyphenated_same_column_words_are_repaired():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="Figure 2. Structure of queries during train-",
            page=1,
            bbox=BoundingBox(309, 620, 545, 631),
        ),
        DocumentBlock(
            type="paragraph",
            text="ing and evaluation.",
            page=1,
            bbox=BoundingBox(309, 631, 377, 642),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 1
    assert result[0].text == "Figure 2. Structure of queries during training and evaluation."


def test_short_centered_math_fragment_does_not_attach_to_prose_paragraph():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="We want the distance to be small for all tasks.",
            page=1,
            bbox=BoundingBox(72, 330, 500, 342),
        ),
        DocumentBlock(
            type="paragraph",
            text="E_{$\\tau$}",
            page=1,
            bbox=BoundingBox(288, 360, 300, 374),
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
        DocumentBlock(
            type="formula",
            text="$2 D(\\phi, W_{\\tau})^{2}$",
            page=1,
            bbox=BoundingBox(308, 360, 368, 376),
            metadata={"formula_mode": "display"},
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert [block.type for block in result] == ["paragraph", "paragraph", "formula"]
    assert result[0].text == "We want the distance to be small for all tasks."


def test_short_diagram_label_does_not_attach_to_prose_line():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="critical technologies and defend essential interests without excessive dependence on another power.",
            page=1,
            bbox=BoundingBox(62, 352, 533, 367),
        ),
        DocumentBlock(
            type="paragraph",
            text="China",
            page=1,
            bbox=BoundingBox(406, 379, 433, 393),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert [block.text for block in result] == [
        "critical technologies and defend essential interests without excessive dependence on another power.",
        "China",
    ]


def test_stacked_pdf_math_fragments_merge_into_one_wide_formula():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="The resulting update for the meta-parameters $\\theta_0$ can be expressed as:",
            page=1,
            bbox=BoundingBox(108, 144, 382, 156),
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
        DocumentBlock(
            type="paragraph",
            text="B",
            page=1,
            bbox=BoundingBox(307, 158, 314, 166),
            metadata={"raw_block_type": "line_with_inline_math", "font_name": "CMMI7", "contains_inline_math": True, "formula_mode": "inline"},
        ),
        DocumentBlock(
            type="paragraph",
            text="X",
            page=1,
            bbox=BoundingBox(303, 166, 318, 176),
            metadata={"raw_block_type": "line_with_inline_math", "font_name": "CMEX10", "contains_inline_math": True, "formula_mode": "inline"},
        ),
        DocumentBlock(
            type="formula",
            text=r"\theta_0 = \theta_0 - \beta\nabla_\theta",
            page=1,
            bbox=BoundingBox(239, 168, 301, 180),
            metadata={"source": "math_zone_detector", "formula_mode": "display", "render_mode": "pdf_crop"},
        ),
        DocumentBlock(
            type="paragraph",
            text=r"L_{Tb}(f_{\theta}b",
            page=1,
            bbox=BoundingBox(319, 168, 351, 180),
            metadata={"raw_block_type": "ambiguous_math_line", "font_name": "CMMI7", "contains_inline_math": True, "formula_mode": "ambiguous"},
        ),
        DocumentBlock(
            type="formula",
            text=r"N(\theta_0))",
            page=1,
            bbox=BoundingBox(348, 168, 373, 182),
            metadata={"source": "math_zone_detector", "formula_mode": "display", "render_mode": "pdf_crop"},
        ),
        DocumentBlock(type="paragraph", text="(3)", page=1, bbox=BoundingBox(492, 168, 504, 179)),
        DocumentBlock(
            type="paragraph",
            text="b = 1",
            page=1,
            bbox=BoundingBox(304, 183, 318, 190),
            metadata={"raw_block_type": "line_with_inline_math", "font_name": "CMMI7", "contains_inline_math": True, "formula_mode": "inline"},
        ),
        DocumentBlock(type="paragraph", text="where beta is a learning rate.", page=1, bbox=BoundingBox(108, 192, 504, 206)),
    ]

    normalized = normalize_math_blocks(blocks)
    rebuilt = rebuild_paragraphs(normalized)
    repaired = repair_display_math_fragments(rebuilt)

    assert repaired[0].type == "paragraph"
    assert not repaired[0].text.endswith("B X")
    assert repaired[1].type == "formula"
    assert repaired[1].metadata["wide_initial_crop"] is True
    assert repaired[1].metadata["merged_formula_fragments"] >= 4
    assert repaired[1].bbox.x0 == 239
    assert repaired[1].bbox.x1 == 504


def test_formula_fragments_skip_semantic_overlap_between_pieces():
    blocks = [
        DocumentBlock(
            type="formula",
            text=r"\theta^{b} i = \theta^{b}",
            page=1,
            bbox=BoundingBox(246, 639, 277, 653),
            metadata={"formula_mode": "display", "render_mode": "pdf_crop"},
        ),
        DocumentBlock(type="paragraph", text="(1)", page=1, bbox=BoundingBox(492, 639, 504, 651)),
        DocumentBlock(
            type="paragraph",
            text=r"), (1) where $\alpha$ is the learning rate",
            page=1,
            bbox=BoundingBox(108, 639, 504, 679),
            metadata={"semantic_only_block": True, "displayable": False, "formula_mode": "display"},
        ),
        DocumentBlock(
            type="formula",
            text=r"i - 1 - \alpha \nabla \theta LS_{b}(f \theta b i_{-1}),",
            page=1,
            bbox=BoundingBox(273, 641, 366, 658),
            metadata={"formula_mode": "display", "render_mode": "pdf_crop"},
        ),
    ]

    repaired = repair_display_math_fragments(blocks)

    assert len(repaired) == 1
    assert repaired[0].type == "formula"
    assert repaired[0].metadata["merged_formula_fragments"] == 3
    assert repaired[0].bbox.x1 == 504


def test_wide_inline_math_residue_does_not_absorb_following_prose():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text=r"_{\} l a_{bel}{_{loss} L_{o_i} = L_{bce} + L_{iou}}",
            page=1,
            bbox=BoundingBox(50, 667, 545, 678),
            metadata={
                "raw_block_type": "line_with_inline_math",
                "font_name": "CMMI10",
                "contains_inline_math": True,
                "formula_mode": "inline",
            },
        ),
        DocumentBlock(
            type="paragraph",
            text="framework. We keep the image encoder of SAM frozen and add adapters.",
            page=1,
            bbox=BoundingBox(62, 681, 286, 696),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 2
    assert result[0].text.startswith(r"_{\} l")
    assert result[1].text.startswith("framework. We keep")


def test_leading_numbered_optimization_residue_is_removed_from_prose():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="(42) minimize $\\phi$ We will show that Reptile corresponds to performing SGD on that objective.",
            page=1,
            bbox=BoundingBox(90, 380, 520, 394),
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 1
    assert result[0].text == "We will show that Reptile corresponds to performing SGD on that objective."


def test_leading_equation_number_is_removed_from_prose_sentence():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="(47) In practice, we cannot exactly compute the projection.",
            page=1,
            bbox=BoundingBox(90, 572, 520, 616),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert len(result) == 1
    assert result[0].text == "In practice, we cannot exactly compute the projection."


def test_split_section_number_is_moved_from_paragraph_tail_to_heading():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="The method is initialized with phi before evaluation. 6",
            page=1,
            bbox=BoundingBox(90, 572, 520, 664),
        ),
        DocumentBlock(
            type="heading",
            text="Experiments",
            page=1,
            bbox=BoundingBox(90, 650, 300, 664),
        ),
    ]

    result = rebuild_paragraphs(blocks)

    assert [block.text for block in result] == [
        "The method is initialized with phi before evaluation.",
        "6 Experiments",
    ]
