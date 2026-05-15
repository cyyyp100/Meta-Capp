import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.chapter_navigation import (
    child_sections_for_chapter,
    extraction_end_page_for_scope,
    heading_titles_match,
    heading_search_start_page,
    normalize_heading_title,
    slice_blocks_for_heading_scope,
    slice_blocks_from_heading_to_end,
)


def test_numbered_children_are_grouped_by_prefix_not_page_range():
    chapters = [
        {"title": "3. Tasks and Motivation", "page_start": 2, "page_end": 2, "toc_level": 1},
        {"title": "3.1. Preliminary", "page_start": 2, "page_end": 2, "toc_level": 2},
        {"title": "4. Meta R-CNN", "page_start": 3, "page_end": 3, "toc_level": 1},
        {"title": "3.2. Few-shot object detection / segmentation", "page_start": 3, "page_end": 3, "toc_level": 2},
        {"title": "4.1. Review the R-CNN family", "page_start": 3, "page_end": 3, "toc_level": 2},
        {"title": "4.2. PRN", "page_start": 4, "page_end": 4, "toc_level": 2},
        {"title": "a question about detector generalization is probably raised:", "page_start": 5, "page_end": 5, "toc_level": 2},
    ]

    children = child_sections_for_chapter(chapters[2], chapters)

    assert [child["title"] for child in children] == [
        "4.1. Review the R-CNN family",
        "4.2. PRN",
    ]


def test_scope_extraction_includes_next_page_to_find_boundary_heading():
    chapters = [
        {"title": "4. Meta R-CNN", "page_start": 3, "page_end": 3, "toc_level": 1},
        {"title": "4.1. Review the R-CNN family", "page_start": 3, "page_end": 3, "toc_level": 2},
        {"title": "4.2. PRN", "page_start": 4, "page_end": 4, "toc_level": 2},
        {"title": "5. Implementation", "page_start": 4, "page_end": 4, "toc_level": 1},
    ]

    assert extraction_end_page_for_scope(chapters[0], chapters, total_pages=12) == 4
    assert extraction_end_page_for_scope(chapters[1], chapters, total_pages=12) == 4


def test_slice_starts_with_selected_subtitle_and_skips_foreign_numbered_section():
    blocks = [
        {"type": "paragraph", "text": "previous page text"},
        {"type": "heading", "level": 1, "text": "4. Meta R-CNN"},
        {"type": "paragraph", "text": "Meta intro"},
        {"type": "heading", "level": 2, "text": "3.2. Few-shot object detection / segmentation"},
        {"type": "paragraph", "text": "foreign subsection text"},
        {"type": "heading", "level": 2, "text": "4.1. Review the R-CNN family"},
        {"type": "paragraph", "text": "R-CNN text"},
        {"type": "heading", "level": 2, "text": "4.2. PRN"},
        {"type": "paragraph", "text": "PRN text"},
        {"type": "heading", "level": 2, "text": "a question about detector generalization is probably raised:"},
        {"type": "heading", "level": 1, "text": "5. Implementation"},
        {"type": "paragraph", "text": "next chapter"},
    ]

    chapter_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": "4. Meta R-CNN", "toc_level": 1},
    )
    subsection_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": "4.1. Review the R-CNN family", "toc_level": 2},
    )

    assert [block["text"] for block in chapter_slice] == [
        "4. Meta R-CNN",
        "Meta intro",
        "4.1. Review the R-CNN family",
        "R-CNN text",
        "4.2. PRN",
        "PRN text",
        "a question about detector generalization is probably raised:",
    ]
    assert chapter_slice[-1]["type"] == "paragraph"
    assert [block["text"] for block in subsection_slice] == [
        "4.1. Review the R-CNN family",
        "R-CNN text",
    ]


def test_slice_keeps_right_column_section_when_previous_subsection_appears_after_it():
    blocks = [
        {
            "type": "heading",
            "level": 1,
            "page_number": 6,
            "bbox": [50, 260, 127, 277],
            "text": "4. Experiments",
        },
        {
            "type": "heading",
            "level": 2,
            "page_number": 6,
            "bbox": [308, 263, 389, 278],
            "text": "4.2. Main Results",
        },
        {
            "type": "heading",
            "level": 2,
            "page_number": 6,
            "bbox": [50, 281, 96, 296],
            "text": "4.1. Setup",
        },
        {
            "type": "paragraph",
            "page_number": 6,
            "bbox": [308, 294, 545, 307],
            "text": "Table 1 shows the few-shot segmentation performance.",
        },
        {
            "type": "paragraph",
            "page_number": 6,
            "bbox": [50, 300, 286, 314],
            "text": "Datasets describe the setup and should not cut 4.2.",
        },
        {
            "type": "paragraph",
            "page_number": 7,
            "bbox": [50, 120, 545, 180],
            "text": "Table 1 reports the main quantitative comparison.",
        },
        {
            "type": "paragraph",
            "page_number": 8,
            "bbox": [50, 120, 545, 180],
            "text": "A float from the next subsection appears before its heading.",
        },
        {
            "type": "heading",
            "level": 2,
            "page_number": 8,
            "bbox": [50, 260, 155, 278],
            "text": "4.3. Ablation Study",
        },
        {"type": "paragraph", "page_number": 8, "text": "Ablation text."},
    ]

    chapter_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": "4.2. Main Results", "page_start": 6, "page_end": 7, "toc_level": 2},
    )

    assert [block["text"] for block in chapter_slice] == [
        "4.2. Main Results",
        "Table 1 shows the few-shot segmentation performance.",
        "Table 1 reports the main quantitative comparison.",
    ]


def test_missing_parent_heading_starts_at_first_child_not_previous_continuation():
    blocks = [
        {"type": "paragraph", "page_number": 3, "text": "objects from the previous chapter"},
        {"type": "paragraph", "page_number": 4, "text": "jects might blend with other classes."},
        {"type": "heading", "page_number": 4, "level": 2, "text": "4.1. Review the R-CNN family"},
        {"type": "paragraph", "page_number": 4, "text": "Faster R-CNN system is known as a two-stage pipeline."},
        {"type": "heading", "page_number": 5, "level": 1, "text": "5. Implementation"},
        {"type": "paragraph", "page_number": 5, "text": "Next chapter."},
    ]

    chapter_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": "4. Meta R-CNN", "page_start": 4, "toc_level": 1},
    )

    assert [block["text"] for block in chapter_slice] == [
        "4.1. Review the R-CNN family",
        "Faster R-CNN system is known as a two-stage pipeline.",
    ]


def test_missing_heading_fallback_drops_leading_page_continuation():
    blocks = [
        {"type": "paragraph", "page_number": 4, "text": "jects might blend with other classes."},
        {"type": "paragraph", "page_number": 4, "text": "Beyond their expectation, we present an intuitive method."},
    ]

    chapter_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": "4. Meta R-CNN", "page_start": 4, "toc_level": 1},
    )

    assert [block["text"] for block in chapter_slice] == [
        "Beyond their expectation, we present an intuitive method.",
    ]


def test_numbered_heading_search_extracts_previous_page():
    assert heading_search_start_page({"title": "4. Meta R-CNN", "page_start": 7}) == 6
    assert heading_search_start_page({"title": "References", "page_start": 7}) == 7


def test_numbered_heading_normalization_ignores_trailing_number_dot():
    assert normalize_heading_title("6.1. Soft Power") == normalize_heading_title("6.1 Soft Power")
    assert normalize_heading_title("3.1. Strategic Autonomy") == normalize_heading_title("3.1 Strategic Autonomy")


def test_numbered_heading_without_space_is_matched():
    blocks = [
        {"type": "heading", "level": 1, "text": "4.Meta R-CNN"},
        {"type": "paragraph", "text": "Chapter intro."},
        {"type": "heading", "level": 1, "text": "5.Implementation"},
    ]

    chapter_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": "4. Meta R-CNN", "toc_level": 1},
    )

    assert [block["text"] for block in chapter_slice] == [
        "4.Meta R-CNN",
        "Chapter intro.",
    ]


def test_overextended_numbered_title_matches_visible_heading_prefix():
    assert heading_titles_match(
        "3.2 Power as a Combination",
        "3.2. Power as a Combination Geopolitical power can be represented with a simple model:",
    )


def test_unnumbered_toc_title_matches_numbered_pdf_heading():
    assert heading_titles_match("5.1 A Simple Climate Equation", "A Simple Climate Equation")


def test_unnumbered_toc_title_starts_at_numbered_heading_not_page_start():
    blocks = [
        {"type": "heading", "level": 1, "page_number": 8, "text": "Chapter 5"},
        {"type": "heading", "level": 1, "page_number": 8, "text": "Environmental Leadership"},
        {"type": "paragraph", "page_number": 8, "text": "Chapter intro."},
        {"type": "heading", "level": 2, "page_number": 8, "text": "5.1 A Simple Climate Equation"},
        {"type": "paragraph", "page_number": 8, "text": "Climate equation content."},
        {"type": "heading", "level": 2, "page_number": 8, "text": "5.2 Difficulties of the Transition"},
        {"type": "paragraph", "page_number": 8, "text": "Transition content."},
    ]

    scoped = slice_blocks_for_heading_scope(
        blocks,
        {"title": "A Simple Climate Equation", "page_start": 8, "page_end": 8, "toc_level": 2},
    )
    from_here = slice_blocks_from_heading_to_end(
        blocks,
        {"title": "A Simple Climate Equation", "page_start": 8, "toc_level": 2},
    )

    assert [block["text"] for block in scoped] == [
        "5.1 A Simple Climate Equation",
        "Climate equation content.",
    ]
    assert [block["text"] for block in from_here][:3] == [
        "5.1 A Simple Climate Equation",
        "Climate equation content.",
        "5.2 Difficulties of the Transition",
    ]


def test_unnumbered_semantic_title_duplicate_does_not_cut_scope_immediately():
    title = "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation"
    blocks = [
        {
            "type": "heading",
            "level": 2,
            "text": title,
            "page_number": 1,
            "metadata": {"displayable": False, "semantic_only_block": True},
        },
        {"type": "heading", "level": 1, "text": "nnU-Net Revisited:", "page_number": 1},
        {"type": "heading", "level": 1, "text": "A Call for Rigorous Validation", "page_number": 1},
        {"type": "paragraph", "text": "in 3D Medical Image Segmentation", "page_number": 1},
        {
            "type": "abstract",
            "text": "Abstract. The release of nnU-Net marked a paradigm shift in medical image segmentation.",
            "page_number": 1,
        },
    ]

    chapter_slice = slice_blocks_for_heading_scope(
        blocks,
        {"title": title, "page_start": 1, "page_end": 1, "toc_level": 1},
    )

    assert [block["text"] for block in chapter_slice] == [
        title,
        "nnU-Net Revisited:",
        "A Call for Rigorous Validation",
        "in 3D Medical Image Segmentation",
        "Abstract. The release of nnU-Net marked a paradigm shift in medical image segmentation.",
    ]


def test_slice_from_heading_to_end_keeps_later_sections():
    blocks = [
        {"type": "heading", "level": 1, "text": "2.1 Economic Power", "page_number": 5},
        {"type": "paragraph", "text": "Selected section text.", "page_number": 5},
        {"type": "heading", "level": 1, "text": "2.2 Political Power", "page_number": 6},
        {"type": "paragraph", "text": "Next section text.", "page_number": 6},
    ]

    chapter_slice = slice_blocks_from_heading_to_end(
        blocks,
        {"title": "2.1 Economic Power", "page_start": 5, "toc_level": 1},
    )

    assert [block["text"] for block in chapter_slice] == [
        "2.1 Economic Power",
        "Selected section text.",
        "2.2 Political Power",
        "Next section text.",
    ]


def test_slice_from_heading_to_end_backfills_same_page_right_column():
    blocks = [
        {
            "id": "right-heading",
            "type": "heading",
            "level": 2,
            "page_number": 3,
            "bbox": [310, 360, 455, 376],
            "metadata": {"page_width": 612},
            "text": "3.2. Few-shot Online Optimizer",
        },
        {
            "id": "right-body",
            "type": "paragraph",
            "page_number": 3,
            "bbox": [310, 390, 545, 440],
            "metadata": {"page_width": 612},
            "text": "Right column text that should be read before page four.",
        },
        {
            "id": "left-heading",
            "type": "heading",
            "level": 2,
            "page_number": 3,
            "bbox": [50, 640, 240, 656],
            "metadata": {"page_width": 612},
            "text": "3.1. Overview of SSM-SAM architecture",
        },
        {
            "id": "left-body",
            "type": "paragraph",
            "page_number": 3,
            "bbox": [50, 666, 286, 702],
            "metadata": {"page_width": 612},
            "text": "Left column text for the selected section.",
        },
        {
            "id": "next-page",
            "type": "paragraph",
            "page_number": 4,
            "bbox": [50, 100, 545, 140],
            "metadata": {"page_width": 612},
            "text": "Page four text.",
        },
    ]

    chapter_slice = slice_blocks_from_heading_to_end(
        blocks,
        {"title": "3.1. Overview of SSM-SAM architecture", "page_start": 3, "toc_level": 2},
    )

    assert [block["id"] for block in chapter_slice] == [
        "left-heading",
        "left-body",
        "right-heading",
        "right-body",
        "next-page",
    ]
