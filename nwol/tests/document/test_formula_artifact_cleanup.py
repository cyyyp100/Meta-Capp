from document.models import BoundingBox, DocumentBlock
from document.postprocess.formula_cropper import _should_crop_formula
from document.postprocess.learning_normalizer import fix_wrong_formula_blocks


def formula(text, x0=300, y0=100, x1=500, y1=120, metadata=None):
    return DocumentBlock(
        type="formula",
        text=text,
        page=1,
        bbox=BoundingBox(x0, y0, x1, y1),
        metadata={"formula_mode": "display", "render_mode": "pdf_crop", **dict(metadata or {})},
        image_path="/tmp/stale_formula.png",
    )


def test_prose_formula_artifact_is_demoted_and_visual_metadata_cleared():
    block = formula("where W^{V} \\in R^{C} \\times^{C} denotes value projection matrix.")

    result = fix_wrong_formula_blocks([block])

    assert result[0].type == "paragraph"
    assert result[0].image_path is None
    assert result[0].metadata["corrected_from"] == "formula"
    assert result[0].metadata["formula_mode"] == "inline"
    assert "formula_image_path" not in result[0].metadata
    assert result[0].metadata.get("render_mode") != "pdf_crop"


def test_formula_cropper_skips_prose_and_citation_fragments():
    assert _should_crop_formula(formula("i = 1 q_{i}k_{i}, has mean 0 and variance d_{k}.")) is False
    assert _should_crop_formula(formula("$ing [62].$")) is False


def test_formula_cropper_keeps_real_display_formula():
    block = formula("Attention(Q, K, V) = softmax(QK^{T} \\\\sqrt{d_{k}})V (1)")

    assert _should_crop_formula(block) is True
