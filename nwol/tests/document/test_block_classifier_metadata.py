import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from document.layout.block_classifier import classify_block
from document.models import BoundingBox, RawBlock, RawLine


def _raw_line(text: str, page: int = 2) -> RawBlock:
    bbox = BoundingBox(108.0, 327.0, 504.0, 338.0)
    return RawBlock(
        text=text,
        bbox=bbox,
        page=page,
        block_type="line",
        lines=[
            RawLine(
                text=text,
                bbox=bbox,
                page=page,
                font_size=10.0,
                font_name="NimbusRomNo9L-Regu",
            )
        ],
    )


def test_citation_line_is_not_author_metadata():
    block = classify_block(
        _raw_line("state of the art results in a variety of settings (Wang et al., 2016; Ba et al., 2016;"),
        body_size=10.0,
    )

    assert block.type == "paragraph"
    assert block.metadata.get("is_metadata") is not True


def test_front_matter_affiliation_is_still_metadata():
    block = classify_block(_raw_line("OpenAI, University of Edinburgh", page=1), body_size=10.0)

    assert block.metadata.get("is_metadata") is True
