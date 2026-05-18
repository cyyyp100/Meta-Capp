import os
import sys
import contextlib
import io
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.document import PDFDocument, normalize_chapter_list
from document.extractors.base import OptionalBackendUnavailable
from document.extractors.noisy import capture_noisy_extractor_output
from document.extractors.opendataloader_extractor import OpenDataLoaderExtractor
from document.extractors.pymupdf_extractor import PyMuPDFExtractor
from document.layout.block_classifier import classify_blocks
from document.layout.column_detector import detect_columns
from document.layout.header_footer import remove_repeated_headers_footers
from document.layout.reading_order import order_page_blocks
from document.models import BoundingBox, DocumentBlock, ExtractionResult, RawBlock, RawLine
from document.pdf_router import clear_cache, extract_document
import document.pdf_router as pdf_router
from document.postprocess.algorithm_figures import _detect_algorithm_regions
from document.postprocess.learning_chunks import enrich_blocks_for_learning
from document.postprocess.figure_extractor import (
    associate_captions,
    blocks_have_missing_managed_assets,
    cleanup_all_document_assets,
    cleanup_document_assets,
    deduplicate_visual_blocks,
    document_asset_dir,
)
from document.postprocess.learning_normalizer import normalize_for_learning
from document.postprocess.inline_formula_repair import repair_same_row_inline_math_fragments
from document.postprocess.list_normalizer import normalize_lists
from document.postprocess.math_normalizer import normalize_math_blocks, normalize_math_text, normalize_unicode_math
from document.postprocess.paragraph_rebuilder import rebuild_paragraphs
from document.postprocess.quality import evaluate_blocks
from document.postprocess.table_normalizer import normalize_tables
from document.postprocess.table_normalizer import _rule_based_table_rects
from document.postprocess.vector_graphics import _graph_drawing_rects


def line(text, page, x0, y0, x1, y1, size=11, bold=False):
    return RawLine(text, BoundingBox(x0, y0, x1, y1), page, size, "Font-Bold" if bold else "Font", bold)


def raw_block(text, page, x0, y0, x1, y1, size=11, bold=False):
    raw_line = line(text, page, x0, y0, x1, y1, size=size, bold=bold)
    return RawBlock(text, raw_line.bbox, page, lines=[raw_line])


def block(text, page=1, x0=60, y0=100, x1=500, y1=115, btype="paragraph"):
    return DocumentBlock(type=btype, text=text, page=page, bbox=BoundingBox(x0, y0, x1, y1))


class _FakeDrawingPage:
    def __init__(self, drawings):
        import fitz

        self._drawings = drawings
        self.rect = fitz.Rect(0, 0, 600, 800)

    def get_drawings(self):
        return self._drawings


def test_repeated_header_footer_removed_without_unique_title():
    lines = []
    page_sizes = {}
    for page in range(1, 5):
        page_sizes[page] = (600, 800)
        lines.append(line(f"Cours analyse page {page}", page, 50, 20, 300, 35))
        lines.append(line(f"{page}", page, 290, 780, 310, 792))
        lines.append(line(f"Corps page {page}", page, 80, 200, 320, 214))
    lines.append(line("Chapitre 1", 1, 80, 55, 250, 75, size=18, bold=True))

    cleaned, removed = remove_repeated_headers_footers(lines, page_sizes)

    texts = [item.text for item in cleaned]
    assert "Chapitre 1" in texts
    assert all("Cours analyse page" not in text for text in texts)
    assert removed


def test_first_page_course_preamble_removed_before_real_heading():
    lines = [
        line("Chapitre 2 : Suites numériques", 1, 110, 20, 470, 35, size=14),
        line("1re-Spécialité mathématiques, 2019-2020", 1, 170, 38, 410, 53, size=14),
        line("1. Mode de génération d’une suite numérique", 1, 25, 88, 425, 105, size=17),
        line("Définition 1.", 1, 32, 136, 100, 147, size=11),
    ]
    page_sizes = {1: (595, 842)}

    cleaned, _ = remove_repeated_headers_footers(lines, page_sizes)

    texts = [item.text for item in cleaned]
    assert texts == ["1. Mode de génération d’une suite numérique", "Définition 1."]


def test_noisy_extractor_stdout_is_captured():
    output = io.StringIO()
    logger = logging.getLogger("test.noisy_extractor")

    with contextlib.redirect_stdout(output):
        with capture_noisy_extractor_output(logger, "Dummy"):
            print("Page 8: could not find the page-dimensions: { ... }")

    assert output.getvalue() == ""


def test_two_column_reading_order_left_then_right():
    blocks = [
        raw_block("right 2", 1, 330, 150, 560, 165),
        raw_block("left 2", 1, 60, 150, 290, 165),
        raw_block("right 1", 1, 330, 90, 560, 105),
        raw_block("left 1", 1, 60, 90, 290, 105),
    ]

    ordered = order_page_blocks(blocks)

    assert [item.text for item in ordered] == ["left 1", "left 2", "right 1", "right 2"]


def test_column_detection_tolerates_caption_and_formula_outliers():
    blocks = [
        block("full caption", page=1, x0=50, y0=195, x1=545, y1=206),
        block("left heading", page=1, x0=50, y0=627, x1=129, y1=642, btype="heading"),
        block("left subheading", page=1, x0=50, y0=647, x1=239, y1=661, btype="heading"),
        block("left body", page=1, x0=62, y0=666, x1=286, y1=678),
        block("right heading", page=1, x0=309, y0=377, x1=456, y1=391, btype="heading"),
        block("right body", page=1, x0=309, y0=397, x1=545, y1=565),
        block("right formula", page=1, x0=381, y0=578, x1=545, y1=619, btype="formula"),
        block("small figure label", page=1, x0=254, y0=92, x1=290, y1=98),
    ]

    layout = detect_columns(blocks, page_width=612)

    assert layout.layout_type == "two_columns"


def test_column_detection_ignores_central_diagram_labels():
    blocks = [
        raw_block("Self-Sampling", 1, 253, 92, 290, 98, size=8),
        raw_block("Transformer layer", 1, 297, 122, 303, 166, size=8),
        raw_block("Adapter", 1, 322, 134, 329, 154, size=8),
        raw_block("Figure 1. Overview of the proposed framework and main components.", 1, 50, 195, 545, 206, size=9),
        raw_block("A caption continuation also spans the full page width.", 1, 50, 207, 545, 218, size=9),
        raw_block("left body first line with enough prose words", 1, 50, 270, 286, 282),
        raw_block("left body second line with enough prose words", 1, 50, 282, 286, 294),
        raw_block("left body third line with enough prose words", 1, 50, 294, 286, 306),
        raw_block("right body first line with enough prose words", 1, 309, 270, 545, 282),
        raw_block("right body second line with enough prose words", 1, 309, 282, 545, 294),
        raw_block("right body third line with enough prose words", 1, 309, 294, 545, 306),
    ]

    layout = detect_columns(blocks, page_width=612)
    ordered = order_page_blocks(blocks, layout)
    texts = [item.text for item in ordered]

    assert layout.layout_type == "two_columns"
    assert texts.index("left body third line with enough prose words") < texts.index(
        "right body first line with enough prose words"
    )


def test_single_column_page_with_short_headings_is_not_split_into_fake_columns():
    blocks = [
        block("4.1 DATASETS", page=7, x0=108, y0=82, x1=176, y1=95, btype="heading"),
        block(
            "The datasets used to evaluate our methods were the Omniglot and Mini-Imagenet datasets.",
            page=7,
            x0=108,
            y0=103,
            x1=504,
            y1=225,
        ),
        block("for testing.", page=7, x0=108, y0=289, x1=151, y1=301),
        block("4.2 EXPERIMENTS", page=7, x0=108, y0=314, x1=193, y1=326, btype="heading"),
        block(
            "To evaluate our methods we adopted a hierarchical hyperparameter search methodology.",
            page=7,
            x0=108,
            y0=334,
            x1=504,
            y1=447,
        ),
        block("5-shot experiments respectively.", page=7, x0=108, y0=565, x1=236, y1=578),
        block("4.3 RESULTS", page=7, x0=108, y0=590, x1=170, y1=603, btype="heading"),
    ]

    layout = detect_columns(blocks, page_width=612)
    ordered = order_page_blocks(blocks, layout)

    assert layout.layout_type == "single_column"
    assert [item.text for item in ordered] == [
        "4.1 DATASETS",
        "The datasets used to evaluate our methods were the Omniglot and Mini-Imagenet datasets.",
        "for testing.",
        "4.2 EXPERIMENTS",
        "To evaluate our methods we adopted a hierarchical hyperparameter search methodology.",
        "5-shot experiments respectively.",
        "4.3 RESULTS",
    ]


def test_single_column_page_with_centered_formula_and_figure_is_not_two_columns():
    blocks = [
        block("Chapter 4", page=7, x0=62, y0=124, x1=164, y1=154, btype="heading"),
        block("Science, Technology and Innovation", page=7, x0=62, y0=169, x1=497, y1=205, btype="heading"),
        block("Europe has a long scientific tradition and remains important in global research.", page=7, x0=62, y0=238, x1=533, y1=281),
        block("computing and biotechnology.", page=7, x0=62, y0=279, x1=205, y1=295),
        block("4.1 Research as an Ecosystem", page=7, x0=62, y0=309, x1=283, y1=330, btype="heading"),
        block("Research depends on universities, public funding, companies and international cooperation.", page=7, x0=62, y0=338, x1=533, y1=455),
        block("Innovation = Research × Education × Investment.", page=7, x0=171, y0=406, x1=424, y1=429, btype="formula"),
        block("", page=7, x0=132, y0=455, x1=462, y1=555, btype="figure"),
        block("Figure 4.1: A simple innovation cycle.", page=7, x0=214, y0=559, x1=381, y1=574),
        block("4.2 The Digital Challenge", page=7, x0=62, y0=592, x1=255, y1=613, btype="heading"),
    ]
    blocks[8].metadata["is_caption"] = True

    layout = detect_columns(blocks, page_width=595)
    ordered = order_page_blocks(blocks, layout)

    assert layout.layout_type == "single_column"
    assert [item.text for item in ordered if item.text] == [
        "Chapter 4",
        "Science, Technology and Innovation",
        "Europe has a long scientific tradition and remains important in global research.",
        "computing and biotechnology.",
        "4.1 Research as an Ecosystem",
        "Research depends on universities, public funding, companies and international cooperation.",
        "Innovation = Research × Education × Investment.",
        "Figure 4.1: A simple innovation cycle.",
        "4.2 The Digital Challenge",
    ]


def test_fig_reference_sentence_is_not_caption():
    classified = classify_blocks([
        raw_block("Fig. 1 depicts the architecture of our few-shot learning", 1, 62, 665, 286, 678),
        raw_block("framework. We keep the image encoder frozen", 1, 50, 678, 286, 690),
    ])

    assert classified[0].metadata.get("is_caption") is not True
    assert classified[1].metadata.get("is_caption") is not True


def test_hyphenated_paragraph_continues_across_column_turn():
    blocks = [
        block("3.1. Overview", page=1, x0=50, y0=647, x1=238, y1=661, btype="heading"),
        block(
            "After pass-",
            page=1,
            x0=50,
            y0=701,
            x1=286,
            y1=714,
        ),
        block(
            "ing through the image encoder, the module updates embeddings.",
            page=1,
            x0=309,
            y0=270,
            x1=545,
            y1=282,
        ),
    ]

    rebuilt = rebuild_paragraphs(blocks, page_sizes={1: (612, 792)})

    assert [item.type for item in rebuilt] == ["heading", "paragraph"]
    assert "After passing through the image encoder" in rebuilt[1].text
    assert rebuilt[1].bbox == blocks[1].bbox
    assert rebuilt[1].metadata["merged_across_columns"] is True


def test_open_paragraph_continues_after_table_interlude():
    blocks = [
        block(
            "In Table 1 we also include the results of our own implementation of MAML. We base our",
            page=7,
            x0=108,
            y0=704,
            x1=504,
            y1=733,
        ),
        DocumentBlock(
            type="paragraph",
            text="Table 1: MAML++ Omniglot 20-way Few-Shot Results.",
            page=8,
            bbox=BoundingBox(108, 89, 504, 101),
            metadata={"is_caption": True, "contains_table": True},
        ),
        DocumentBlock(
            type="table",
            text="Approach | 1-shot | 5-shot",
            page=8,
            bbox=BoundingBox(175, 156, 434, 350),
        ),
        block(
            "conclusions on the relative performance between our own MAML implementation and the proposed methodologies.",
            page=8,
            x0=108,
            y0=627,
            x1=504,
            y1=650,
        ),
    ]

    rebuilt = rebuild_paragraphs(blocks, page_sizes={7: (612, 792), 8: (612, 792)})

    assert rebuilt[0].type == "paragraph"
    assert "We base our conclusions on the relative performance" in rebuilt[0].text
    assert rebuilt[0].metadata["merged_across_visual_interlude"] is True
    assert rebuilt[0].metadata["merged_across_pages"] is True
    assert [item.type for item in rebuilt[1:]] == ["paragraph", "table"]


def test_pymupdf_line_spans_split_when_columns_are_merged():
    class Rect:
        width = 595

    class Page:
        rect = Rect()

        def get_text(self, _kind):
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": [50, 100, 545, 112],
                        "lines": [
                            {
                                "bbox": [50, 100, 545, 112],
                                "spans": [
                                    {"text": "left column first words", "bbox": [50, 100, 286, 112], "size": 10, "font": "Times"},
                                    {"text": "right column first words", "bbox": [310, 100, 545, 112], "size": 10, "font": "Times"},
                                ],
                            }
                        ],
                    }
                ]
            }

    lines = PyMuPDFExtractor()._extract_page_lines(Page(), 1)

    assert [item.text for item in lines] == ["left column first words", "right column first words"]
    assert lines[0].bbox.x1 < 300
    assert lines[1].bbox.x0 > 300


def test_router_fusion_preserves_two_column_reading_order():
    geo = ExtractionResult(
        blocks=[
            DocumentBlock("paragraph", text="left 1", page=1, bbox=BoundingBox(60, 90, 290, 105)),
            DocumentBlock("paragraph", text="left 2", page=1, bbox=BoundingBox(60, 150, 290, 165)),
            DocumentBlock("paragraph", text="right 1", page=1, bbox=BoundingBox(330, 90, 560, 105)),
            DocumentBlock("paragraph", text="right 2", page=1, bbox=BoundingBox(330, 150, 560, 165)),
        ],
        pages=1,
        score=0.9,
        warnings=[],
        engine_name="pymupdf_structured",
        debug_paths=[],
    )

    fused = pdf_router._fuse_geometric_and_semantic_results(
        geo,
        [],
        document_type="scientific_article_two_columns",
    )

    assert [item.text for item in fused.blocks] == ["left 1", "left 2", "right 1", "right 2"]


def test_heading_subheading_and_paragraph_classification():
    raw = [
        raw_block("Chapitre 1", 1, 60, 50, 260, 75, size=22, bold=True),
        raw_block("1.1 Sous-titre", 1, 60, 95, 260, 112, size=15, bold=True),
        raw_block("Un paragraphe de cours normal avec assez de texte.", 1, 60, 130, 500, 145, size=11),
        raw_block("Deuxieme ligne de corps.", 1, 60, 150, 500, 165, size=11),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "heading"
    assert classified[0].level == 1
    assert classified[1].type == "heading"
    assert classified[1].level == 2
    assert classified[2].type == "paragraph"


def test_single_level_dotted_section_heading_is_detected():
    raw = [
        raw_block("1. Introduction", 1, 60, 95, 210, 112, size=15),
        raw_block("Un paragraphe de corps normal.", 1, 60, 130, 500, 145, size=11),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "heading"
    assert classified[0].level == 1
    assert classified[0].text == "1. Introduction"


def test_numbered_heading_without_space_is_detected():
    raw = [
        raw_block("3.2Few-shot object detection / segmentation", 1, 60, 95, 360, 112, size=15),
        raw_block("Un paragraphe de corps normal.", 1, 60, 130, 500, 145, size=11),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "heading"
    assert classified[0].level == 2


def test_appendix_letter_heading_and_embedded_body_are_split():
    raw = [
        raw_block("E. Extended Results", 21, 60, 95, 310, 116, size=18, bold=True),
        raw_block(
            "E.1. Main Results We include detailed numbers corresponding to figures in the main body.",
            21,
            80,
            135,
            560,
            152,
            size=14,
            bold=True,
        ),
    ]

    classified = classify_blocks(raw)

    assert [block.type for block in classified] == ["heading", "heading", "paragraph"]
    assert classified[0].level == 1
    assert classified[1].level == 2
    assert classified[1].text == "E.1. Main Results"
    assert classified[2].text.startswith("We include detailed numbers")


def test_abstract_heading_is_kept_apart_then_merged_semantically():
    raw = [
        raw_block("Abstract", 1, 150, 95, 220, 112, size=15),
        raw_block("This paper introduces a robust method.", 1, 60, 130, 500, 145, size=11),
        raw_block("1. Introduction", 1, 60, 170, 210, 187, size=15),
    ]

    rebuilt = rebuild_paragraphs(classify_blocks(raw))
    normalized = normalize_for_learning(rebuilt)

    assert [block.type for block in rebuilt] == ["heading", "paragraph", "heading"]
    assert normalized[0].type == "abstract"
    assert normalized[0].text.startswith("Abstract This paper")
    assert normalized[1].type == "heading"


def test_side_metadata_is_not_promoted_to_heading_or_merged():
    raw = [
        raw_block("Title", 1, 60, 50, 260, 75, size=22, bold=True),
        raw_block("arXiv:1909.13032v2 [cs.CV] 14 Mar 2020", 1, 10, 200, 38, 560, size=20),
        raw_block("Abstract", 1, 150, 95, 220, 112, size=15),
    ]

    rebuilt = rebuild_paragraphs(classify_blocks(raw))

    assert rebuilt[1].type == "paragraph"
    assert rebuilt[1].metadata.get("is_metadata") is True
    assert rebuilt[1].text.startswith("arXiv:")
    assert rebuilt[2].text == "Abstract"


def test_author_affiliations_are_marked_as_metadata():
    raw = [
        raw_block(
            "Tianang Leng* Huazhong University of Science and Technology Wuhan, China tianangl@hust.edu.cn",
            1,
            60,
            90,
            520,
            118,
            size=13,
            bold=True,
        ),
        raw_block(
            "Kun Han University of California, Irvine Irvine, California, United States khan7@uci.edu",
            1,
            60,
            135,
            520,
            165,
            size=13,
        ),
    ]

    classified = classify_blocks(raw)

    assert [item.type for item in classified] == ["paragraph", "paragraph"]
    assert all(item.metadata.get("is_metadata") is True for item in classified)


def test_late_content_with_center_is_not_author_metadata():
    raw = [
        raw_block(
            "Europe’s place in the world is complex. It is no longer the dominant center of global power, but it",
            10,
            60,
            240,
            530,
            255,
            size=11,
        ),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "paragraph"
    assert classified[0].metadata.get("is_metadata") is not True


def test_late_geographic_list_is_not_author_metadata():
    raw = [
        raw_block(
            "Paris, Rome, Barcelona, Amsterdam, Vienna and Prague attract millions of visitors because they",
            9,
            60,
            240,
            530,
            255,
            size=11,
        ),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "paragraph"
    assert classified[0].metadata.get("is_metadata") is not True


def test_learning_enrichment_makes_front_matter_metadata_non_interactive():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="Kun Han University of California, Irvine Irvine, California, United States khan7@uci.edu",
            page=1,
            bbox=BoundingBox(60, 135, 520, 165),
            id="p1_b4",
        )
    ]

    enriched = enrich_blocks_for_learning(blocks, document_type="scientific_article")

    assert enriched[0].metadata["is_metadata"] is True
    assert enriched[0].metadata["displayable"] is False
    assert enriched[0].metadata["interactive"] is False


def test_learning_enrichment_stops_reference_state_at_additional_results():
    blocks = [
        DocumentBlock(type="heading", text="REFERENCES", page=10, bbox=BoundingBox(108, 80, 200, 95)),
        DocumentBlock(
            type="paragraph",
            text="Alex Author. A cited paper. arXiv preprint arXiv:1605.00000, 2016.",
            page=10,
            bbox=BoundingBox(108, 100, 504, 114),
        ),
        DocumentBlock(type="heading", text="ADDITIONAL RESULTS", page=10, bbox=BoundingBox(129, 522, 245, 537)),
        DocumentBlock(
            type="table",
            text="Inner Loop Steps | 1 | 2",
            page=10,
            bbox=BoundingBox(134, 625, 478, 659),
            metadata={"is_reference": True},
        ),
    ]

    enriched = enrich_blocks_for_learning(blocks, document_type="scientific_article")

    assert enriched[1].metadata["is_reference"] is True
    assert enriched[2].metadata.get("is_reference") is not True
    assert enriched[3].metadata.get("is_reference") is not True


def test_learning_enrichment_marks_lowercase_caption_continuation():
    blocks = [
        DocumentBlock(
            type="paragraph",
            text="parameter, per layer, per step for LSLR and (f times l times (s -1))",
            page=11,
            bbox=BoundingBox(108, 351, 504, 370),
            metadata={"is_caption": True},
        ),
        DocumentBlock(
            type="paragraph",
            text="layers preceding batch normalization, l is number of layers and s is number of inner loop step.",
            page=11,
            bbox=BoundingBox(108, 362, 504, 385),
        ),
    ]

    enriched = enrich_blocks_for_learning(blocks, document_type="scientific_article")

    assert enriched[1].metadata["is_caption"] is True
    assert enriched[1].metadata["displayable"] is False


def test_learning_enrichment_suppresses_covered_semantic_list_duplicate():
    text = (
        "Our proposed methodologies improve the original MAML framework. "
        "Table 1 one can see how the approach performs on Omniglot. "
        "The method improves convergence and stability."
    )
    blocks = [
        DocumentBlock(
            type="paragraph",
            text=text,
            page=7,
            bbox=BoundingBox(108, 611, 504, 704),
            id="geo",
        ),
        DocumentBlock(
            type="bullet_list",
            text=f"• {text}",
            items=[text],
            page=7,
            bbox=BoundingBox(108, 622, 504, 733),
            id="semantic",
            metadata={"semantic_only_block": True},
        ),
    ]

    enriched = enrich_blocks_for_learning(blocks, document_type="scientific_article_two_columns")

    assert enriched[1].metadata["displayable"] is False
    assert enriched[1].metadata["interactive"] is False
    assert enriched[1].metadata["suppressed_semantic_duplicate"] is True


def test_learning_enrichment_suppresses_duplicate_against_cross_page_anchor():
    anchor_text = (
        "We base our conclusions on the relative performance between our own "
        "MAML implementation and the proposed methodologies. Table 2 showcases "
        "the Mini-Imagenet tasks."
    )
    blocks = [
        DocumentBlock(
            type="paragraph",
            text=anchor_text,
            page=7,
            bbox=BoundingBox(108, 704, 504, 733),
            id="anchor",
            metadata={"page_start": 7, "page_end": 8, "merged_across_pages": True},
        ),
        DocumentBlock(
            type="bullet_list",
            text=(
                "• Table 2 showcases the Mini-Imagenet tasks and summarizes "
                "the relative performance between the proposed methodologies."
            ),
            items=[
                (
                    "Table 2 showcases the Mini-Imagenet tasks and summarizes "
                    "the relative performance between the proposed methodologies."
                )
            ],
            page=8,
            bbox=BoundingBox(108, 654, 504, 733),
            id="semantic",
            metadata={"semantic_only_block": True},
        ),
    ]

    enriched = enrich_blocks_for_learning(blocks, document_type="scientific_article_two_columns")

    assert enriched[1].metadata["displayable"] is False
    assert enriched[1].metadata["suppressed_semantic_duplicate"] is True


def test_segmentation_preserves_metadata_flags_for_cached_pages():
    from core.segmentation import segment_blocks

    segmented = segment_blocks([
        {
            "type": "paragraph",
            "text": "Tianang Leng* Huazhong University of Science and Technology Wuhan, China tianangl@hust.edu.cn",
            "metadata": {},
        }
    ])

    assert segmented[0]["is_metadata"] is True
    assert segmented[0]["metadata"]["is_metadata"] is True


def test_captions_and_cross_references_are_not_promoted_to_headings():
    raw = [
        raw_block("Figure2. U-Net architecture.", 1, 60, 95, 320, 112, size=14, bold=True),
        raw_block("Section 2.3 describes the U-Net architecture.", 1, 60, 130, 500, 145, size=14, bold=True),
        raw_block("2.3. U-Net", 1, 60, 170, 210, 187, size=15, bold=True),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "paragraph"
    assert classified[0].metadata.get("is_caption") is True
    assert classified[1].type == "paragraph"
    assert classified[2].type == "heading"


def test_table_reference_sentence_is_not_marked_as_caption():
    classified = classify_blocks([
        raw_block("Table 1 one can see how our method performs on Omniglot.", 7, 108, 622, 504, 634),
        raw_block("Table 1: MAML++ Omniglot 20-way Few-Shot Results.", 8, 108, 90, 504, 102),
    ])

    assert classified[0].metadata.get("is_caption") is not True
    assert classified[1].metadata.get("is_caption") is True


def test_methodologies_sentence_fragment_is_not_promoted_to_heading():
    classified = classify_blocks([
        raw_block("methodologies.", 8, 108, 638, 169, 650, size=10),
    ])

    assert classified[0].type == "paragraph"


def test_semantic_callout_line_with_body_is_not_rendered_as_heading():
    raw = [
        raw_block(
            "Définition 1.1.1 (Suites équivalentes). Soient u_n et v_n deux suites.",
            1,
            60,
            95,
            520,
            112,
            size=14,
            bold=True,
        ),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "definition"


def test_discourse_numbered_sentence_is_not_promoted_to_heading():
    blocks = normalize_for_learning([
        DocumentBlock(
            type="paragraph",
            text="8.2. First, we find that object attentive vectors tend to cluster.",
            page=1,
        )
    ])

    assert blocks[0].type == "paragraph"


def test_math_zone_detector_ignores_hyphenated_prose_labels():
    from document.layout.math_zone_detector import looks_like_math_fragment

    assert looks_like_math_fragment(line("X-Ray", 1, 10, 10, 50, 20)) is False
    assert looks_like_math_fragment(line("2.3. U-Net", 1, 10, 30, 90, 40)) is False
    assert looks_like_math_fragment(line("[PubMed]", 1, 10, 50, 80, 60)) is False


def test_split_numbered_heading_fragments_are_merged():
    raw = [
        raw_block("1.1", 1, 70, 100, 92, 120, size=14, bold=True),
        raw_block("Étude locale", 1, 108, 100, 196, 120, size=14, bold=True),
    ]

    classified = classify_blocks(raw)

    assert len(classified) == 1
    assert classified[0].type == "heading"
    assert classified[0].level == 2
    assert classified[0].text == "1.1 Étude locale"


def test_uppercase_numbered_heading_fragments_are_merged_without_bold_signal():
    raw = [
        raw_block("3.1", 4, 108, 232, 122, 244, size=10),
        raw_block("MODEL AGNOSTIC META-LEARNING PROBLEMS", 4, 132, 232, 336, 244, size=10),
        raw_block("The simplicity, elegance and high performance of MAML make it powerful.", 4, 108, 252, 504, 264, size=10),
    ]

    classified = classify_blocks(raw)

    assert classified[0].type == "heading"
    assert classified[0].level == 2
    assert classified[0].text == "3.1 MODEL AGNOSTIC META-LEARNING PROBLEMS"
    assert classified[1].type == "paragraph"


def test_learning_normalizer_repairs_plain_text_formulas_and_callouts():
    blocks = normalize_for_learning(
        [
            block("$2.3. U-Net$", btype="formula"),
            block("$[CrossRef] [PubMed]$", y0=130, y1=144, btype="formula"),
            DocumentBlock(type="heading", text="Remarque 2", page=1, bbox=BoundingBox(60, 160, 130, 174)),
            block("La réciproque est fausse.", y0=178, y1=192),
        ]
    )

    assert blocks[0].type == "heading"
    assert blocks[0].level == 2
    assert blocks[0].text == "2.3. U-Net"
    assert blocks[1].type == "paragraph"
    assert blocks[1].text == "[CrossRef] [PubMed]"
    assert blocks[2].type == "remark"
    assert "La réciproque est fausse." in blocks[2].text


def test_learning_normalizer_promotes_opendataloader_numbered_paragraph_heading():
    blocks = normalize_for_learning(
        [
            block("3.2.Few-shot object detection / segmentation", page=4),
            block("Novel-class objects are discussed here.", page=4, y0=130, y1=144),
        ]
    )

    assert blocks[0].type == "heading"
    assert blocks[0].level == 2
    assert blocks[0].text == "3.2. Few-shot object detection / segmentation"
    assert blocks[0].metadata["promoted_from"] == "paragraph"


def test_learning_normalizer_splits_opendataloader_heading_joined_to_body():
    blocks = normalize_for_learning(
        [
            block(
                "3.2.Few-shot object detection / segmentation In this setting, novel classes have few examples.",
                page=4,
            ),
        ]
    )

    assert [item.type for item in blocks] == ["heading", "paragraph"]
    assert blocks[0].text == "3.2. Few-shot object detection / segmentation"
    assert blocks[0].level == 2
    assert blocks[1].text.startswith("In this setting")


def test_learning_normalizer_splits_opendataloader_bullet_heading_joined_to_body():
    blocks = normalize_for_learning(
        [
            DocumentBlock(
                type="bullet_list",
                text="• 3.2. Few-shot object detection / segmentation From visual recognition to detection.",
                page=3,
                items=["3.2. Few-shot object detection / segmentation From visual recognition to detection."],
            ),
        ]
    )

    assert [item.type for item in blocks] == ["heading", "paragraph"]
    assert blocks[0].text == "3.2. Few-shot object detection / segmentation"
    assert blocks[0].level == 2
    assert blocks[0].metadata["promoted_from"] == "bullet_list"
    assert blocks[1].text.startswith("From visual recognition")


def test_callout_heading_a_retenir_is_not_merged_with_previous_paragraph():
    raw = [
        raw_block("Autrement dit, u_n et v_n ont le même comportement. À retenir", 1, 60, 130, 500, 145, size=11),
        raw_block("Écrire u_n ∼ v_n revient à écrire", 1, 60, 170, 300, 185, size=11),
    ]

    classified = classify_blocks(raw)
    rebuilt = rebuild_paragraphs(classified)

    assert classified[1].type == "heading"
    assert classified[1].text == "À retenir"
    assert [item.type for item in rebuilt] == ["paragraph", "heading", "paragraph"]


def test_opendataloader_overextended_heading_is_split_before_chapter_indexing():
    blocks = normalize_for_learning(
        [
            DocumentBlock(
                type="heading",
                text="3.2 Power as a Combination Geopolitical power can be represented with a simple model:",
                page=6,
                level=5,
                metadata={"source": "opendataloader_pdf"},
            )
        ]
    )

    assert [(block.type, block.text) for block in blocks] == [
        ("heading", "3.2. Power as a Combination"),
        ("paragraph", "Geopolitical power can be represented with a simple model:"),
    ]
    assert blocks[0].level == 2
    assert blocks[0].metadata["split_embedded_heading_body"] is True


def test_pdf_toc_keeps_subsubsections():
    doc = PDFDocument("/tmp/cours.pdf")
    doc.page_count = 12
    doc.toc = [
        {"level": 1, "title": "Chapitre 1", "page": 1},
        {"level": 2, "title": "1.1 Sous-chapitre", "page": 2},
        {"level": 3, "title": "1.1.1 Sous-sous-titre", "page": 3},
        {"level": 3, "title": "1.1.2 Suite", "page": 5},
        {"level": 2, "title": "1.2 Autre", "page": 7},
        {"level": 1, "title": "Chapitre 2", "page": 10},
    ]

    chapters = doc._toc_to_chapters()

    assert [item["toc_level"] for item in chapters] == [1, 2, 3, 3, 2, 1]
    assert chapters[0]["page_end"] == 9
    assert chapters[1]["page_end"] == 6
    assert chapters[2]["page_end"] == 4


def test_pdf_document_can_rebuild_toc_from_extracted_headings():
    doc = PDFDocument("/tmp/paper.pdf")
    doc.page_count = 35
    doc.chapters = doc._make_pseudo_chapters()
    blocks = [
        DocumentBlock(type="heading", text="Universeg", page=1, level=1),
        DocumentBlock(type="heading", text="MIT CSAIL", page=1, level=3),
        DocumentBlock(type="heading", text="A. Method", page=11, level=1),
        DocumentBlock(type="heading", text="A.1. Architecture", page=12, level=2),
        DocumentBlock(type="heading", text="34.9. Figure 5 also shows clear qualitative improvements", page=14, level=2),
        DocumentBlock(type="heading", text="References", page=14, level=3),
        DocumentBlock(type="heading", text="E. Extended Results", page=21, level=1),
        DocumentBlock(type="heading", text="E.1. Main Results", page=21, level=2),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    assert [chapter["title"] for chapter in doc.chapters] == [
        "A. Method",
        "A.1. Architecture",
        "References",
        "E. Extended Results",
        "E.1. Main Results",
    ]
    assert [chapter["toc_level"] for chapter in doc.chapters] == [1, 2, 1, 1, 2]


def test_pdf_document_rebuilds_numbered_heading_levels_when_backend_offsets_them():
    doc = PDFDocument("/tmp/metarcnn.pdf")
    doc.page_count = 12
    doc.chapters = doc._make_pseudo_chapters()
    blocks = [
        DocumentBlock(type="heading", text="1. Introduction", page=1, level=2),
        DocumentBlock(type="heading", text="2. Related Work", page=2, level=2),
        DocumentBlock(type="heading", text="3. Tasks and Motivation", page=3, level=2),
        DocumentBlock(type="heading", text="3.1. Preliminary", page=3, level=3),
        DocumentBlock(type="heading", text="3.2. Few-shot object detection / segmentation", page=4, level=3),
        DocumentBlock(type="heading", text="References", page=12, level=3),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    top_level_titles = [chapter["title"] for chapter in doc.chapters if chapter["toc_level"] == 1]
    assert top_level_titles == [
        "1. Introduction",
        "2. Related Work",
        "3. Tasks and Motivation",
        "References",
    ]
    assert [chapter["toc_level"] for chapter in doc.chapters] == [1, 1, 1, 2, 2, 1]


def test_pdf_document_enriches_native_toc_with_missing_extracted_subheading():
    doc = PDFDocument("/tmp/metarcnn.pdf")
    doc.page_count = 12
    doc.has_toc = True
    doc.chapters = [
        {"title": "3. Tasks and Motivation", "page_start": 3, "page_end": 5, "toc_level": 1},
        {"title": "3.1. Preliminary", "page_start": 3, "page_end": 5, "toc_level": 2},
        {"title": "4. Meta R-CNN", "page_start": 5, "page_end": 11, "toc_level": 1},
    ]
    blocks = [
        DocumentBlock(type="heading", text="3.1. Preliminary", page=3, level=2),
        DocumentBlock(type="heading", text="3.2.Few-shot object detection / segmentation", page=4, level=2),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    assert [chapter["title"] for chapter in doc.chapters] == [
        "3. Tasks and Motivation",
        "3.1. Preliminary",
        "3.2. Few-shot object detection / segmentation",
        "4. Meta R-CNN",
    ]
    assert [chapter["toc_level"] for chapter in doc.chapters] == [1, 2, 2, 1]


def test_pdf_document_deduplicates_native_and_numbered_extracted_toc_entries():
    doc = PDFDocument("/tmp/article.pdf")
    doc.page_count = 8
    doc.has_toc = True
    doc.chapters = [
        {"title": "Introduction", "page_start": 1, "page_end": 2, "toc_level": 1},
        {"title": "Methodology", "page_start": 3, "page_end": 6, "toc_level": 1},
        {"title": ". Experiments", "page_start": 7, "page_end": 8, "toc_level": 1},
    ]
    blocks = [
        DocumentBlock(type="heading", text="1. Introduction", page=1, level=1),
        DocumentBlock(type="heading", text="2. Methodology", page=3, level=1),
        DocumentBlock(type="heading", text="3. Experiments", page=7, level=1),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    assert [chapter["title"] for chapter in doc.chapters] == [
        "1. Introduction",
        "2. Methodology",
        "3. Experiments",
    ]


def test_normalize_chapter_list_drops_native_article_title_before_numbered_sections():
    chapters = normalize_chapter_list(
        [
            {
                "title": "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation",
                "page_start": 1,
                "page_end": 1,
                "toc_level": 1,
            },
            {
                "title": "1. Introduction",
                "page_start": 1,
                "page_end": 2,
                "toc_level": 1,
            },
            {
                "title": "2. Methods",
                "page_start": 3,
                "page_end": 5,
                "toc_level": 1,
            },
        ],
        page_count=6,
    )

    assert [chapter["title"] for chapter in chapters] == [
        "1. Introduction",
        "2. Methods",
    ]


def test_normalize_chapter_list_drops_front_matter_metadata_and_pseudo_when_real_headings_exist():
    chapters = normalize_chapter_list(
        [
            {
                "title": "1. Introduction",
                "page_start": 1,
                "page_end": 2,
                "toc_level": 1,
            },
            {
                "title": "arXiv:2304.06131v1[cs.CV]12Apr2023",
                "page_start": 1,
                "page_end": 1,
                "toc_level": 1,
            },
            {
                "title": "Pages 1–10",
                "page_start": 1,
                "page_end": 10,
                "toc_level": 1,
            },
            {
                "title": "2. Related Works",
                "page_start": 2,
                "page_end": 4,
                "toc_level": 1,
            },
        ],
        page_count=10,
    )

    assert [chapter["title"] for chapter in chapters] == [
        "1. Introduction",
        "2. Related Works",
    ]


def test_normalize_chapter_list_trims_opendataloader_heading_body_artifact():
    chapters = normalize_chapter_list(
        [
            {
                "title": "3.2. Power as a Combination Geopolitical power can be represented with a simple model:",
                "page_start": 6,
                "toc_level": 2,
            },
        ],
        page_count=10,
    )

    assert chapters[0]["title"] == "3.2. Power as a Combination"


def test_normalize_chapter_list_drops_opendataloader_contents_page_artifacts():
    chapters = normalize_chapter_list(
        [
            {
                "title": "The Place of Europe in the World",
                "page_start": 1,
                "toc_level": 2,
            },
            {
                "title": "Cyprien Vial",
                "page_start": 1,
                "toc_level": 3,
            },
            {
                "title": "Preface",
                "page_start": 2,
                "toc_level": 1,
            },
            {
                "title": "Contents",
                "page_start": 3,
                "toc_level": 1,
            },
            {
                "title": "1 Historical Influence of Europe 1",
                "page_start": 3,
                "toc_level": 4,
            },
            {
                "title": "7 Conclusion 7",
                "page_start": 3,
                "toc_level": 4,
            },
            {
                "title": "Historical Influence of Europe",
                "page_start": 4,
                "toc_level": 1,
            },
            {
                "title": "1.1 The European Heritage",
                "page_start": 4,
                "toc_level": 3,
            },
            {
                "title": "The European Union",
                "page_start": 5,
                "toc_level": 1,
            },
            {
                "title": "2.1 Economic Power",
                "page_start": 5,
                "toc_level": 3,
            },
        ],
    )

    titles = [chapter["title"] for chapter in chapters]
    level1_titles = [chapter["title"] for chapter in chapters if chapter["toc_level"] == 1]

    assert "Contents" not in titles
    assert "1. Historical Influence of Europe 1" not in titles
    assert "7. Conclusion 7" not in titles
    assert "Historical Influence of Europe" in level1_titles
    assert "The European Union" in level1_titles


def test_normalize_chapter_list_keeps_unnumbered_chapter_with_numbered_subsections():
    chapters = normalize_chapter_list(
        [
            {
                "title": "Méthode générale de résolution des équations différentielles linéaires",
                "page_start": 1,
                "page_end": 4,
                "toc_level": 1,
            },
            {
                "title": "1.1. Cas homogène",
                "page_start": 1,
                "page_end": 2,
                "toc_level": 2,
            },
            {
                "title": "1.2. Second membre",
                "page_start": 3,
                "page_end": 4,
                "toc_level": 2,
            },
        ],
        page_count=4,
    )

    assert "Méthode générale de résolution des équations différentielles linéaires" in [
        chapter["title"] for chapter in chapters
    ]


def test_pdf_document_normalizes_spaced_course_numbering_levels():
    doc = PDFDocument("/tmp/course.pdf")
    doc.page_count = 4
    doc.chapters = doc._make_pseudo_chapters()
    blocks = [
        DocumentBlock(type="heading", text="1. Mode de génération", page=1, level=1),
        DocumentBlock(type="heading", text="1. 1. Définition d’une suite", page=1, level=1),
        DocumentBlock(type="heading", text="1. 2. Relation de récurrence", page=2, level=1),
        DocumentBlock(type="heading", text="2. Suites arithmétiques", page=3, level=1),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    assert [chapter["title"] for chapter in doc.chapters] == [
        "1. Mode de génération",
        "1.1. Définition d’une suite",
        "1.2. Relation de récurrence",
        "2. Suites arithmétiques",
    ]
    assert [chapter["toc_level"] for chapter in doc.chapters] == [1, 2, 2, 1]


def test_pdf_document_ignores_section_cross_references_as_chapters():
    doc = PDFDocument("/tmp/thesis.pdf")
    doc.page_count = 25
    doc.chapters = doc._make_pseudo_chapters()
    blocks = [
        DocumentBlock(type="heading", text="1. Introduction", page=1, level=1),
        DocumentBlock(type="heading", text="Section 2.3 describes the U-Net architecture", page=2, level=2),
        DocumentBlock(type="heading", text="2. Theoretical Background", page=3, level=1),
        DocumentBlock(type="heading", text="2.1. Medical Image Segmentation", page=3, level=2),
        DocumentBlock(type="heading", text="2.1. First, we find that object vectors tend to cluster", page=4, level=2),
        DocumentBlock(type="heading", text="References", page=20, level=1),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    assert [chapter["title"] for chapter in doc.chapters] == [
        "1. Introduction",
        "2. Theoretical Background",
        "2.1. Medical Image Segmentation",
        "References",
    ]


def test_pdf_document_ignores_metadata_heading_when_rebuilding_toc():
    doc = PDFDocument("/tmp/universeg.pdf")
    doc.page_count = 35
    doc.chapters = doc._make_pseudo_chapters()
    blocks = [
        DocumentBlock(
            type="heading",
            text="arXiv:2304.06131v1[cs.CV]12Apr2023",
            page=1,
            level=1,
            metadata={"is_metadata": True},
        ),
        DocumentBlock(type="heading", text="1. Introduction", page=1, level=1),
        DocumentBlock(type="heading", text="2. Related Works", page=2, level=1),
    ]

    changed = doc.update_chapters_from_blocks(blocks)

    assert changed is True
    assert [chapter["title"] for chapter in doc.chapters] == [
        "1. Introduction",
        "2. Related Works",
    ]


def test_multiline_bullet_list():
    blocks = [
        block("- premier item", y0=100, y1=114),
        block("suite du premier item", x0=85, y0=116, y1=130),
        block("2) second item", y0=150, y1=164),
    ]

    result = normalize_lists(blocks)

    assert len(result) == 1
    assert result[0].type == "bullet_list"
    assert result[0].items == ["premier item suite du premier item", "second item"]
    assert "• second item" in result[0].text


def test_inline_bullet_markers_are_split_inside_pdf_line():
    result = normalize_lists(
        [
            block(
                "• La suite u est aussi notée $(u_{n})_{n}$. •$u_{n+1}$ est le terme suivant.",
                y0=100,
                y1=114,
            ),
        ]
    )

    assert len(result) == 1
    assert result[0].type == "bullet_list"
    assert result[0].items == [
        "La suite u est aussi notée $(u_{n})_{n}$.",
        "$u_{n+1}$ est le terme suivant.",
    ]
    assert "\n• $u_{n+1}$" in result[0].text


def test_numbered_course_section_is_not_normalized_as_bullet_list():
    result = normalize_lists(
        [
            block("1. 1. Définition d’une suite numérique", y0=100, y1=114),
            block("Définition 1.", y0=130, y1=144),
        ]
    )

    assert [item.type for item in result] == ["paragraph", "paragraph"]


def test_isolated_math_formula_block():
    result = normalize_math_blocks([block("∀x∈R, x^2 ≥ 0")])

    assert result[0].type == "formula"
    assert result[0].latex == r"\forall x \in R, x^2 \geq 0"


def test_unicode_math_normalization_repairs_combining_not_equal_and_indices():
    assert normalize_unicode_math("uₙ ̸= 0") == r"u_n \neq 0"
    assert normalize_unicode_math("uₙ ̸ = 0") == r"u_n \neq 0"

    result = normalize_math_blocks([block("uₙ ̸= 0")])

    assert result[0].type == "formula"
    assert result[0].text == r"$u_n \neq 0$"
    assert result[0].latex == r"u_n \neq 0"


def test_math_span_joining_keeps_spaces_around_inline_formula():
    extractor = PyMuPDFExtractor()
    spans = [
        {"text": "Écrire", "bbox": [0, 0, 34, 10], "size": 11, "font": "Times"},
        {"text": "u", "bbox": [40, 0, 46, 10], "size": 11, "font": "CMMI"},
        {"text": "n", "bbox": [47, 5, 51, 10], "size": 7, "font": "CMMI"},
        {"text": "∼", "bbox": [56, 0, 62, 10], "size": 11, "font": "CMSY"},
        {"text": "v", "bbox": [67, 0, 73, 10], "size": 11, "font": "CMMI"},
        {"text": "n", "bbox": [74, 5, 78, 10], "size": 7, "font": "CMMI"},
        {"text": "revient", "bbox": [84, 0, 124, 10], "size": 11, "font": "Times"},
        {"text": "à écrire", "bbox": [130, 0, 174, 10], "size": 11, "font": "Times"},
    ]

    assert extractor._join_spans(spans) == "Écrire u_{n}∼v_{n} revient à écrire"


def test_same_visual_line_inline_math_fragments_do_not_become_formula_crops():
    from document.layout.math_zone_detector import raw_lines_to_math_aware_blocks

    lines = [
        RawLine("Where the projections are parameter matrices W^{Q}", BoundingBox(107.5, 203.1, 303.0, 215.7), 1, 10, "Times"),
        RawLine("i", BoundingBox(295.3, 210.6, 298.1, 217.6), 1, 7, "CMMI7"),
        RawLine("∈R^{d}model^{×}^{d}k, W^{K}", BoundingBox(306.2, 203.3, 377.3, 215.7), 1, 10, "CMMI7"),
        RawLine("i", BoundingBox(369.2, 210.5, 372.1, 217.4), 1, 7, "CMMI7"),
        RawLine("∈R^{d}model^{×}^{d}k, W^{V}", BoundingBox(381.1, 203.3, 450.2, 215.7), 1, 10, "CMMI7"),
        RawLine("i", BoundingBox(444.1, 210.5, 446.9, 217.4), 1, 7, "CMMI7"),
        RawLine("∈R^{d}model^{×}^{d}v", BoundingBox(455.1, 203.3, 502.8, 215.4), 1, 10, "CMMI7"),
        RawLine("and W^{O}∈R^{hd}v^{×}^{d}model.", BoundingBox(108.0, 215.6, 201.2, 227.9), 1, 10, "Times"),
    ]

    raw_blocks = raw_lines_to_math_aware_blocks(lines, {1: (612.0, 792.0)})
    classified = classify_blocks(raw_blocks)
    repaired = repair_same_row_inline_math_fragments(classified)
    normalized = normalize_math_blocks(repaired)

    local = [block for block in normalized if block.bbox and 200 <= block.bbox.y0 <= 230]
    assert local
    assert all(block.type == "paragraph" for block in local)

    rebuilt = rebuild_paragraphs(normalized, page_sizes={1: (612.0, 792.0)})
    text = " ".join(block.text for block in rebuilt)
    assert "Where the projections are parameter matrices" in text
    assert "and" in text


def test_mixed_prose_math_stays_paragraph():
    result = normalize_math_blocks([block("Écrire u_n ∼ v_n revient à écrire")])

    assert result[0].type == "paragraph"
    assert result[0].text == r"Écrire $u_n \sim v_n$ revient à écrire"


def test_glued_mixed_prose_math_is_repaired_as_paragraph():
    result = normalize_math_blocks([block("Écrireu_n∼v_nrevientàécrire")])

    assert result[0].type == "paragraph"
    assert result[0].text == r"Écrire $u_n \sim v_n$ revient à écrire"


def test_split_equivalence_formula_fragments_are_rebuilt():
    fragments = [
        "uₙ ∼ vₙ",
        "(n → +∞),",
        "si",
        "uₙ",
        "−−−−−→",
        "n→+∞",
        "1.",
        "vₙ",
    ]
    blocks = [
        block(text, y0=100 + index * 16, y1=112 + index * 16)
        for index, text in enumerate(fragments)
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].type == "formula"
    assert result[0].latex == (
        r"u_n \sim v_n (n \rightarrow + \infty), "
        r"\quad \mathrm{si} \quad "
        r"\frac{u_n}{v_n}\rightarrow_{n \rightarrow + \infty}1."
    )


def test_standalone_limit_subscripts_are_repaired_before_rendering():
    blocks = [
        block("u_n", y0=100, y1=112),
        block("----- →", y0=116, y1=128),
        block(r"_{n}_{\rightarrow}_{+}_{\infty}1.", y0=132, y1=144),
        block("v_n", y0=148, y1=160),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].type == "formula"
    assert result[0].latex == r"\frac{u_n}{v_n}\rightarrow_{n \rightarrow + \infty}1."


def test_split_limit_expression_fragments_are_rebuilt():
    blocks = [
        block("Écrire u_n ∼ v_n revient à écrire", y0=100, y1=112),
        block("u_n = v_n(1 + εn), εn −−−−−→", y0=116, y1=128),
        block("n→+∞", y0=132, y1=144),
        block("0,", y0=148, y1=160),
        block("ou encore", y0=180, y1=192),
        block("u_n = v_n + o(v_n).", y0=196, y1=208),
    ]

    result = normalize_math_blocks(blocks)

    assert [item.type for item in result] == ["paragraph", "formula", "paragraph", "formula"]
    assert result[1].latex == (
        r"u_n = v_n(1 + \epsilon_{n}), "
        r"\epsilon_{n}\rightarrow_{n \rightarrow + \infty}0,"
    )
    assert result[3].latex == r"u_n = v_n + o(v_n)."


def test_leading_limit_arrow_fragment_is_reordered_with_equation_line():
    blocks = [
        block("εn −−−−−→", y0=100, y1=112),
        block("u_n = v_n(1 + εn),", y0=116, y1=128),
        block("n→+∞0,", y0=132, y1=144),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].latex == (
        r"u_n = v_n(1 + \epsilon_{n}), "
        r"\epsilon_{n}\rightarrow_{n \rightarrow + \infty}0,"
    )


def test_orphan_math_prefix_is_removed_before_prose():
    result = normalize_math_blocks(
        [block(". n Comme n^2 - n ~ n^2 et ln(1 + 1 / n) ~ 1 / n, on obtient")]
    )

    assert result[0].type == "paragraph"
    assert result[0].text.startswith("Comme")
    assert not result[0].text.startswith(". n")
    assert r"$n^2 - n \sim n^2$" in result[0].text
    assert r"$\ln(1 + 1 / n) \sim 1 / n$" in result[0].text


def test_inline_relation_expressions_are_wrapped_as_groups():
    result = normalize_math_blocks(
        [block("Comme n^2 - n ~ n^2 et ln(1 + 1 / n) ~ 1 / n, on obtient")]
    )

    assert result[0].text == (
        r"Comme $n^2 - n \sim n^2$ et "
        r"$\ln(1 + 1 / n) \sim 1 / n$, on obtient"
    )


def test_split_unit_fraction_fragments_are_rebuilt():
    blocks = [
        block("u_n ~ n^2 · 1", y0=100, y1=112),
        block("n = n.", y0=132, y1=144),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].type == "formula"
    assert result[0].latex == r"u_n \sim n^2\cdot \frac{1}{n} = n."


def test_broken_negative_exponential_superscript_is_repaired():
    result = normalize_math_blocks(
        [
            block("u_{n} = (n + 1)e^{-}^{n}.", y0=100, y1=112),
            block(r"u_{n} \sim ne^{-}^{n}.", y0=132, y1=144),
        ]
    )

    assert [item.type for item in result] == ["formula", "formula"]
    assert result[0].latex == r"u_{n} = (n + 1)e^{-n}."
    assert result[1].latex == r"u_{n} \sim ne^{-n}."
    assert all("^{-}^" not in item.latex for item in result)


def test_general_unicode_math_symbols_and_script_runs_are_supported():
    result = normalize_math_blocks(
        [
            block("∀x∈ℝ, x² ≥ 0", y0=100, y1=112),
            block("x⁻¹ + y₁₂", y0=132, y1=144),
        ]
    )

    assert [item.type for item in result] == ["formula", "formula"]
    assert result[0].latex == r"\forall x \in \mathbb{R}, x^2 \geq 0"
    assert result[1].latex == r"x^{-1} + y_{12}"


def test_general_functions_limits_roots_and_fractions_are_repaired():
    result = normalize_math_blocks(
        [
            block("lim_{x->0} sin x / x = 1", y0=100, y1=112),
            block("rank(A) ≤ min(n,p)", y0=132, y1=144),
            block("sqrt(x^2 + 1)", y0=164, y1=176),
            block("u_n = 1 / n", y0=196, y1=208),
        ]
    )

    assert [item.type for item in result] == ["formula", "formula", "formula", "formula"]
    assert result[0].latex == r"\lim_{x \rightarrow 0} \frac{\sin x}{x} = 1"
    assert result[1].latex == r"\mathrm{rank}(A) \leq \min(n,p)"
    assert result[2].latex == r"\sqrt{x^2 + 1}"
    assert result[3].latex == r"u_n = \frac{1}{n}"


def test_generic_stacked_fraction_with_bar_is_rebuilt():
    blocks = [
        block("x^2 + 1", y0=100, y1=110, x0=120, x1=180),
        block("-----", y0=113, y1=118, x0=116, x1=184),
        block("x - 1", y0=121, y1=131, x0=125, x1=175),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].type == "formula"
    assert result[0].latex == r"\frac{x^2 + 1}{x - 1}"


def test_generic_formula_continuation_is_joined():
    blocks = [
        block("f(x) =", y0=100, y1=112),
        block("x^2 + 1", y0=116, y1=128),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].latex == r"f(x) = x^2 + 1"


def test_orphan_formula_garbage_fragments_are_removed():
    blocks = [
        block(") (", y0=100, y1=112),
        block("1 + 1", y0=128, y1=140),
        block("u_n = (n^2 - n)/n", y0=164, y1=176),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].type == "formula"
    assert result[0].latex == r"u_n = \frac{(n^2 - n)}{n}"


def test_truncated_latex_command_fragments_are_removed():
    blocks = [
        block(r"\tex", y0=100, y1=112, btype="formula"),
        block(r"\text", y0=128, y1=140, btype="formula"),
        block("u_n = 1", y0=164, y1=176, btype="formula"),
    ]

    result = normalize_math_blocks(blocks)

    assert len(result) == 1
    assert result[0].latex == "u_n = 1"


def test_figure_caption_association():
    figure = DocumentBlock(
        type="figure",
        image_path="assets/doc/page_1_img_1.png",
        page=1,
        bbox=BoundingBox(100, 100, 400, 300),
    )
    caption = block("Figure 1 : Courbe de convergence", y0=310, y1=325)
    text = block("Paragraphe suivant", y0=350, y1=365)

    result = associate_captions([caption, text], [figure])

    figures = [item for item in result if item.type == "figure"]
    assert figures[0].caption == "Figure 1 : Courbe de convergence"
    assert caption not in result
    assert text in result


def test_multiline_caption_repairs_soft_hyphenation():
    figure = DocumentBlock(
        type="figure",
        image_path="assets/doc/page_1_img_1.png",
        page=1,
        bbox=BoundingBox(100, 100, 400, 300),
    )
    caption_head = block("Figure 2. Structure of queries during train-", y0=310, y1=321)
    caption_tail = block("ing and evaluation.", y0=322, y1=333)
    caption_tail.metadata["is_caption"] = True

    result = associate_captions([caption_head, caption_tail], [figure])

    figures = [item for item in result if item.type == "figure"]
    assert figures[0].caption == "Figure 2. Structure of queries during training and evaluation."
    assert caption_head not in result
    assert caption_tail not in result


def test_shared_caption_for_image_group_is_kept_once_for_display():
    upper = DocumentBlock(
        type="figure",
        image_path="assets/doc/page_1_img_1.png",
        page=1,
        bbox=BoundingBox(100, 100, 360, 220),
    )
    lower = DocumentBlock(
        type="figure",
        image_path="assets/doc/page_1_img_2.png",
        page=1,
        bbox=BoundingBox(100, 260, 360, 380),
    )
    caption = block("Figure 1. Different medical imaging modalities [19].", y0=230, y1=245)

    result = associate_captions([caption], [upper, lower], max_distance=80)

    figures = [item for item in result if item.type == "figure"]
    assert [item.caption for item in figures] == [caption.text, caption.text]
    assert figures[0].metadata["caption_display"] is True
    assert figures[1].metadata["caption_display"] is False
    assert caption not in result


def test_table_caption_association_preserves_table_content():
    table = DocumentBlock(
        type="table",
        text="Sector | Example\nEnergy | Gas",
        markdown="| Sector | Example |\n| --- | --- |\n| Energy | Gas |",
        page=1,
        bbox=BoundingBox(100, 130, 400, 260),
    )
    caption = block("Table 2.1: Examples of important European economic sectors.", y0=104, y1=118)

    result = associate_captions([caption], [table], max_distance=80)

    tables = [item for item in result if item.type == "table"]
    assert tables[0].caption == caption.text
    assert tables[0].text == "Sector | Example\nEnergy | Gas"
    assert tables[0].metadata["caption_display"] is True
    assert caption not in result


def test_rule_based_table_rects_detect_horizontal_rule_table():
    import fitz

    drawings = [
        {"type": "s", "items": [("l",)] * 1, "rect": fitz.Rect(100.3, 482.94, 494.98, 482.94)},
        {"type": "s", "items": [("l",)] * 1, "rect": fitz.Rect(100.3, 502.12, 494.98, 502.12)},
        {"type": "s", "items": [("l",)] * 1, "rect": fitz.Rect(100.3, 575.50, 494.98, 575.50)},
    ]

    rects = _rule_based_table_rects(_FakeDrawingPage(drawings))

    assert len(rects) == 1
    assert abs(rects[0].x0 - 100.3) < 0.01
    assert abs(rects[0].y1 - 575.50) < 0.01


def test_orphan_table_caption_is_not_reader_visible():
    from core.segmentation import segment_blocks

    segmented = segment_blocks([
        {
            "type": "paragraph",
            "text": "Table 2.1: Examples of important European economic sectors.",
            "metadata": {"is_caption": True},
        }
    ])

    assert segmented == []


def test_duplicate_visual_figures_are_collapsed_and_metadata_is_preserved():
    native = DocumentBlock(
        type="figure",
        image_path="assets/doc/native.png",
        page=1,
        bbox=BoundingBox(100, 100, 400, 300),
        confidence=0.85,
        metadata={"source": "pdf_native_image"},
    )
    schema_crop = DocumentBlock(
        type="figure",
        image_path="assets/doc/schema.png",
        caption="Figure 2. Architecture",
        text="Figure 2. Architecture",
        page=1,
        bbox=BoundingBox(102, 98, 402, 302),
        confidence=0.72,
        metadata={
            "source": "vector_graphic_drawing",
            "contains_schema": True,
            "llm_assets": [{"type": "image", "path": "assets/doc/schema.png", "reason": "vector_graphic"}],
        },
    )
    paragraph = block("Texte suivant", y0=350, y1=365)

    result = deduplicate_visual_blocks([native, schema_crop, paragraph])
    figures = [item for item in result if item.type == "figure"]

    assert len(figures) == 1
    assert figures[0].caption == "Figure 2. Architecture"
    assert figures[0].metadata["contains_schema"] is True
    assert figures[0].metadata["deduplicated_visual"] is True
    assert figures[0].metadata["llm_assets"][0]["path"] == "assets/doc/schema.png"
    assert paragraph in result


def test_subfigures_inside_composite_are_not_deduplicated_by_containment_only():
    composite = DocumentBlock(
        type="figure",
        image_path="assets/doc/composite.png",
        page=1,
        bbox=BoundingBox(50, 80, 550, 420),
    )
    subfigure = DocumentBlock(
        type="figure",
        image_path="assets/doc/subfigure.png",
        page=1,
        bbox=BoundingBox(80, 110, 260, 220),
    )

    result = deduplicate_visual_blocks([composite, subfigure])

    assert len([item for item in result if item.type == "figure"]) == 2


def test_document_assets_cleanup_removes_only_document_dir(tmp_path):
    pdf_path = tmp_path / "cours.pdf"
    pdf_path.write_text("pdf")
    root = tmp_path / "assets"
    asset_dir = document_asset_dir(pdf_path, output_root=root)
    asset_dir.mkdir(parents=True)
    image_path = asset_dir / "page_1_img_1.png"
    image_path.write_bytes(b"png")

    assert not blocks_have_missing_managed_assets([{"type": "figure", "image_path": str(image_path)}])

    removed = cleanup_document_assets(pdf_path, output_root=root)

    assert removed == 1
    assert not asset_dir.exists()
    assert blocks_have_missing_managed_assets([{"type": "figure", "image_path": str(image_path)}])


def test_cleanup_all_document_assets_removes_asset_cache(tmp_path):
    root = tmp_path / "assets"
    kept_root = root / "doc_a" / "context"
    kept_root.mkdir(parents=True)
    image_path = kept_root / "context_p1_0.png"
    image_path.write_bytes(b"png")
    other_path = root / "doc_b" / "page_1_img_1.png"
    other_path.parent.mkdir(parents=True)
    other_path.write_bytes(b"png")

    removed = cleanup_all_document_assets(output_root=root)

    assert removed == 2
    assert root.exists()
    assert not any(root.iterdir())


def test_missing_context_asset_invalidates_cached_blocks(tmp_path):
    pdf_path = tmp_path / "cours.pdf"
    pdf_path.write_text("pdf")
    root = tmp_path / "assets"
    asset_dir = document_asset_dir(pdf_path, output_root=root) / "context"
    asset_dir.mkdir(parents=True)
    image_path = asset_dir / "context_p1_0.png"
    image_path.write_bytes(b"png")
    block = {
        "type": "paragraph",
        "metadata": {
            "context_asset_path": str(image_path),
            "llm_assets": [{"type": "image", "path": str(image_path), "reason": "inline_math"}],
        },
    }

    assert not blocks_have_missing_managed_assets([block])

    image_path.unlink()

    assert blocks_have_missing_managed_assets([block])


def test_vector_graphics_skip_filled_text_panel_but_keep_compact_diagram():
    import fitz

    drawings = [
        {
            "type": "f",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(70, 100, 524, 220),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(70, 250, 168, 278),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(203, 250, 301, 278),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(336, 250, 434, 278),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(469, 250, 567, 278),
        },
    ]

    rects = _graph_drawing_rects(_FakeDrawingPage(drawings))

    assert len(rects) == 1
    assert rects[0].y0 == 250
    assert rects[0].x0 == 70
    assert rects[0].x1 == 567


def test_vector_graphics_merges_spaced_compact_flow_diagram():
    import fitz

    drawings = [
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(127.16, 507.86, 206.53, 536.21),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(257.95, 507.86, 337.32, 536.21),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(388.75, 507.86, 468.12, 536.21),
        },
        {
            "type": "fs",
            "items": [("rect",)] * 8,
            "rect": fitz.Rect(239.53, 587.63, 355.75, 615.98),
        },
    ]

    rects = _graph_drawing_rects(_FakeDrawingPage(drawings))

    assert len(rects) == 1
    assert abs(rects[0].x0 - 127.16) < 0.01
    assert abs(rects[0].x1 - 468.12) < 0.01
    assert abs(rects[0].y1 - 615.98) < 0.01


def test_algorithm_regions_are_detected_as_panel():
    import fitz

    class FakeAlgorithmPage:
        rect = fitz.Rect(0, 0, 612, 792)

        def get_text(self, _mode):
            return [
                (50, 72, 222, 85, "Algorithm 1 SSM-SAM Online Optimizer", 0, 0),
                (54, 88, 286, 113, "Input: K image-mask pairs of the target organ", 0, 0),
                (55, 124, 286, 149, "Require: learning rate alpha; number of steps S", 0, 0),
                (56, 160, 170, 187, "2: for s from 1 to S do", 0, 0),
                (56, 221, 98, 234, "6: end for", 0, 0),
                (50, 286, 286, 370, "We employ meta-learning for our model training.", 0, 0),
            ]

    regions = _detect_algorithm_regions(FakeAlgorithmPage())

    assert len(regions) == 1
    rect, title = regions[0]
    assert title == "Algorithm 1 SSM-SAM Online Optimizer"
    assert rect.y0 == 72
    assert rect.y1 == 234


def test_simple_table_normalization():
    result = normalize_tables(
        [
            block("Nom  Valeur  Unite", y0=100, y1=114),
            block("a    1       m", y0=116, y1=130),
            block("b    2       s", y0=132, y1=146),
        ]
    )

    assert len(result) == 1
    assert result[0].type == "table"
    assert "| Nom | Valeur | Unite |" in result[0].markdown
    assert "<table>" in result[0].html


def test_long_paragraph_split_cleanly():
    text = " ".join("Phrase courte." for _ in range(120))

    result = rebuild_paragraphs([block(text)], max_chars=700)

    assert len(result) > 1
    assert all(len(item.text) <= 760 for item in result)


def test_long_paragraph_split_does_not_cut_inline_math():
    formula = "$" + " + ".join(f"x_{index}.y_{index}" for index in range(30)) + "$"
    text = f"Avant la formule. {formula} Après la formule, le paragraphe continue."

    result = rebuild_paragraphs([block(text)], max_chars=70)

    assert len(result) >= 3
    assert all(item.text.count("$") != 1 for item in result)
    assert any(item.text.startswith("$") and item.text.endswith("$") for item in result)


def test_glued_math_symbol_spacing():
    assert normalize_math_text("Ωappelé espace fondamental") == r"\Omega appelé espace fondamental"


def test_quality_penalizes_glued_and_long_blocks():
    score, warnings = evaluate_blocks([block("Ωappelé " + "x" * 1300)], pages=4)

    assert score < 1.0
    assert warnings


def test_router_does_not_crash_when_optional_backends_absent(monkeypatch):
    clear_cache()

    def fake_pymupdf_extract(self, pdf_path):
        return ExtractionResult(
            blocks=[DocumentBlock(type="paragraph", text="Texte minimal", page=1)],
            pages=1,
            score=0.2,
            warnings=["score faible"],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf_extract)

    result = extract_document("/tmp/missing.pdf")

    assert result.engine_name == "pymupdf_structured"
    assert result.blocks
    clear_cache()


def test_router_returns_formula_blocks(monkeypatch):
    clear_cache()

    def fake_pymupdf_extract(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(type="formula", text="E=mc^2", latex="E=mc^2", page=1),
                DocumentBlock(type="paragraph", text="Suite du cours.", page=1),
            ],
            pages=1,
            score=0.9,
            warnings=[],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf_extract)

    result = extract_document("/tmp/missing.pdf")

    assert result.engine_name == "pymupdf_structured"
    assert any(b.type == "formula" for b in result.blocks)
    assert any(b.type == "paragraph" for b in result.blocks)
    clear_cache()


def test_learning_enrichment_marks_displayable_and_multimodal_visual_route():
    paragraph = DocumentBlock(
        type="paragraph",
        text=(
            "Figure 1 shows the proposed architecture and explains how the visual "
            "modules interact with the text encoder during inference."
        ),
        page=1,
        bbox=BoundingBox(60, 100, 500, 150),
        id="p1_b0",
    )
    figure = DocumentBlock(
        type="figure",
        text="Figure 1. Proposed architecture.",
        caption="Figure 1. Proposed architecture.",
        image_path="/tmp/schema.png",
        page=1,
        bbox=BoundingBox(70, 170, 480, 390),
        id="p1_b1",
    )

    enriched = enrich_blocks_for_learning([paragraph, figure], document_type="scientific_article")

    assert enriched[0].metadata["displayable"] is True
    assert enriched[0].metadata["interactive"] is True
    assert enriched[0].metadata["generation_mode"] == "llm_multimodal"
    assert enriched[0].metadata["visual_assets"][0]["image_path"] == "/tmp/schema.png"
    assert enriched[1].metadata["attached_to_block_id"] == enriched[0].id


def test_algorithm_visual_stays_standalone_during_learning_enrichment():
    paragraph = DocumentBlock(
        type="paragraph",
        text=(
            "The online optimizer adapts the segmentation model to each target organ "
            "using a small set of annotated images."
        ),
        page=4,
        bbox=BoundingBox(310, 100, 545, 155),
        id="p4_text",
    )
    algorithm = DocumentBlock(
        type="figure",
        text="Algorithm 1 SSM-SAM Online Optimizer",
        caption="Algorithm 1 SSM-SAM Online Optimizer",
        image_path="/tmp/algorithm.png",
        page=4,
        bbox=BoundingBox(50, 70, 296, 252),
        id="p4_algorithm",
        metadata={
            "source": "algorithm_text_panel",
            "contains_schema": True,
            "contains_algorithm": True,
        },
    )

    enriched = enrich_blocks_for_learning([paragraph, algorithm], document_type="scientific_article_two_columns")

    assert "visual_assets" not in enriched[0].metadata
    assert "llm_assets" not in enriched[0].metadata
    assert "attached_to_block_id" not in enriched[1].metadata
    assert enriched[1].metadata["visual_assets"][0]["image_path"] == "/tmp/algorithm.png"
    assert enriched[1].metadata["generation_mode"] == "llm_multimodal"


def test_router_fuses_semantic_text_onto_pymupdf_geometry(monkeypatch):
    clear_cache()

    def fake_pymupdf_extract(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="paragraph",
                    text=(
                        "Abstract Introduction Methods Results Conclusion References "
                        "This paper introduces a robust reptile method for language model training."
                    ),
                    page=1,
                    bbox=BoundingBox(60, 100, 500, 170),
                    confidence=0.86,
                )
            ],
            pages=1,
            score=0.9,
            warnings=[],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    def fake_opendataloader_extract(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="paragraph",
                    text=(
                        "Abstract Introduction Methods Results Conclusion References "
                        "This paper introduces a robust Reptile method for language model training "
                        "with cleaner wording."
                    ),
                    page=1,
                    bbox=BoundingBox(62, 101, 502, 171),
                    confidence=0.95,
                    metadata={"source": "opendataloader_pdf"},
                )
            ],
            pages=1,
            score=0.92,
            warnings=[],
            engine_name="opendataloader_pdf",
            debug_paths=[],
        )

    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf_extract)
    monkeypatch.setattr(OpenDataLoaderExtractor, "extract", fake_opendataloader_extract)
    monkeypatch.setattr(
        pdf_router,
        "_try_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Marker must not run automatically")),
    )

    result = extract_document("/tmp/missing.pdf")

    assert result.engine_name == "pymupdf_structured+opendataloader_pdf"
    assert result.metadata["document_type"] == "scientific_article"
    assert result.blocks[0].bbox == BoundingBox(60, 100, 500, 170)
    assert "cleaner wording" in result.blocks[0].text
    assert result.blocks[0].metadata["text_enriched_by"] == "opendataloader_pdf"
    assert result.blocks[0].metadata["displayable"] is True

    clear_cache()
    scientific = extract_document("/tmp/missing.pdf", preferred_engine="scientific")

    assert scientific.engine_name == "pymupdf_structured+opendataloader_pdf"
    assert "cleaner wording" in scientific.blocks[0].text
    clear_cache()


def test_router_upgrades_paragraph_to_heading_when_semantic_confirms(monkeypatch):
    """A PyMuPDF paragraph gets upgraded to heading when OpenDataLoader explicitly tags it as heading."""
    clear_cache()

    def fake_pymupdf(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="paragraph",
                    text="Conclusion",
                    page=2,
                    bbox=BoundingBox(60, 100, 300, 118),
                    confidence=0.9,
                ),
                DocumentBlock(
                    type="paragraph",
                    text="This paper showed that the proposed method works well in all tested scenarios.",
                    page=2,
                    bbox=BoundingBox(60, 130, 500, 145),
                    confidence=0.9,
                ),
            ],
            pages=2,
            score=0.9,
            warnings=[],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    def fake_odl(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="heading",
                    text="Conclusion",
                    page=2,
                    bbox=BoundingBox(62, 101, 302, 119),
                    level=1,
                    confidence=1.0,
                    metadata={"source": "opendataloader_pdf"},
                ),
            ],
            pages=2,
            score=0.9,
            warnings=[],
            engine_name="opendataloader_pdf",
            debug_paths=[],
        )

    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf)
    monkeypatch.setattr(OpenDataLoaderExtractor, "extract", fake_odl)

    result = extract_document("/tmp/missing.pdf")

    heading_blocks = [b for b in result.blocks if b.type == "heading"]
    assert len(heading_blocks) == 1
    assert heading_blocks[0].text == "Conclusion"
    assert heading_blocks[0].level == 1
    assert heading_blocks[0].metadata.get("type_upgraded_to_heading_by") == "opendataloader_pdf"

    para_blocks = [b for b in result.blocks if b.type == "paragraph"]
    assert any(b.metadata.get("parent_title") == "Conclusion" for b in para_blocks)

    clear_cache()


def test_router_does_not_upgrade_paragraph_when_semantic_heading_contains_embedded_body(monkeypatch):
    clear_cache()

    def fake_pymupdf(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="heading",
                    text="3.2 Power as a Combination",
                    page=6,
                    bbox=BoundingBox(60, 100, 280, 120),
                    level=2,
                    confidence=0.9,
                ),
                DocumentBlock(
                    type="paragraph",
                    text="Geopolitical power can be represented with a simple model: China competition",
                    page=6,
                    bbox=BoundingBox(60, 130, 360, 146),
                    confidence=0.9,
                ),
            ],
            pages=6,
            score=0.9,
            warnings=[],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    def fake_odl(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="heading",
                    text="3.2 Power as a Combination Geopolitical power can be represented with a simple model:",
                    page=6,
                    bbox=BoundingBox(62, 101, 362, 146),
                    level=5,
                    confidence=1.0,
                    metadata={"source": "opendataloader_pdf"},
                ),
            ],
            pages=6,
            score=0.9,
            warnings=[],
            engine_name="opendataloader_pdf",
            debug_paths=[],
        )

    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf)
    monkeypatch.setattr(OpenDataLoaderExtractor, "extract", fake_odl)

    result = extract_document("/tmp/missing.pdf")

    texts_by_type = [(block.type, block.text) for block in result.blocks]
    assert ("heading", "3.2 Power as a Combination") in texts_by_type
    assert ("paragraph", "Geopolitical power can be represented with a simple model: China competition") in texts_by_type
    assert all(
        block.text != "3.2 Power as a Combination Geopolitical power can be represented with a simple model:"
        for block in result.blocks
    )

    clear_cache()


def test_router_does_not_upgrade_long_paragraph_to_heading(monkeypatch):
    """A long paragraph must NOT be upgraded even if OpenDataLoader tags it as heading (ODL error guard)."""
    clear_cache()

    long_text = "This is a very long paragraph that exceeds the heading length guard. " * 4

    def fake_pymupdf(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="paragraph",
                    text=long_text,
                    page=1,
                    bbox=BoundingBox(60, 100, 500, 145),
                    confidence=0.9,
                ),
            ],
            pages=1,
            score=0.9,
            warnings=[],
            engine_name="pymupdf_structured",
            debug_paths=[],
        )

    def fake_odl(self, pdf_path):
        return ExtractionResult(
            blocks=[
                DocumentBlock(
                    type="heading",
                    text=long_text,
                    page=1,
                    bbox=BoundingBox(62, 101, 502, 146),
                    level=1,
                    confidence=1.0,
                    metadata={"source": "opendataloader_pdf"},
                ),
            ],
            pages=1,
            score=0.9,
            warnings=[],
            engine_name="opendataloader_pdf",
            debug_paths=[],
        )

    monkeypatch.setattr(PyMuPDFExtractor, "extract", fake_pymupdf)
    monkeypatch.setattr(OpenDataLoaderExtractor, "extract", fake_odl)

    result = extract_document("/tmp/missing.pdf")

    para_blocks = [b for b in result.blocks if b.type == "paragraph"]
    heading_blocks = [b for b in result.blocks if b.type == "heading"]
    assert len(para_blocks) >= 1
    assert len(heading_blocks) == 0

    clear_cache()
