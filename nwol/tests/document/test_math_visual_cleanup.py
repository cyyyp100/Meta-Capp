from document.models import BoundingBox, DocumentBlock
from document.postprocess.context_assets import _is_unsafe_inline_math_crop_geometry, _needs_context_asset
from document.postprocess.math_visual_cleanup import cleanup_visual_math_fragments


def block(text, x0, y0, x1, y1, btype="paragraph", metadata=None):
    return DocumentBlock(
        type=btype,
        text=text,
        page=1,
        bbox=BoundingBox(x0, y0, x1, y1),
        metadata=dict(metadata or {}),
    )


def test_visual_math_residue_is_absorbed_into_crop_owner():
    blocks = [
        block(
            "Donc $u_{n} \\rightarrow 0$, d'ou $q^{n} = o$(n!). Enfin, $n_{n} = 1$ n!",
            70,
            120,
            286,
            180,
            metadata={"formula_mode": "inline", "contains_inline_math": True},
        ),
        block("n $\\cdot$ 2", 280, 150, 303, 188),
        block("$n \\cdot... \\cdot n$", 297, 154, 341, 188, btype="formula"),
        block(
            "n. Pour $n \\geq 2$, au moins la moitie des facteurs est inferieure a 1 / 2.",
            70,
            162,
            524,
            210,
            metadata={"formula_mode": "inline", "contains_inline_math": True},
        ),
    ]

    cleaned = cleanup_visual_math_fragments(blocks)

    assert len(cleaned) == 2
    assert cleaned[0].metadata["visual_math_fragment_group"] is True
    assert cleaned[0].metadata["render_mode"] == "context_crop_only"
    assert cleaned[0].bbox.x1 == 341
    assert cleaned[1].text.startswith("Pour $n \\geq 2$")
    assert cleaned[1].metadata["stripped_leading_formula_residue"] is True


def test_plain_math_paragraph_is_not_forced_to_crop_only_without_residue():
    blocks = [
        block(
            "Comme $n^{2} + n \\sim n^{2}$, le terme dominant est clair.",
            70,
            120,
            520,
            140,
            metadata={"formula_mode": "inline", "contains_inline_math": True},
        )
    ]

    cleaned = cleanup_visual_math_fragments(blocks)

    assert cleaned == blocks


def test_external_metadata_does_not_become_context_crop():
    metadata_block = block(
        "https: / / doi.org / 10.3390 / diagnostics14121213",
        35,
        540,
        115,
        550,
    )

    assert _needs_context_asset(metadata_block) is False


def test_wide_mixed_column_inline_math_crop_is_unsafe():
    inline_math_block = block(
        "$L_{o_i} = L_{bce} + L_{iou}$ framework. We keep the encoder frozen.",
        50,
        667,
        545,
        714,
        metadata={
            "contains_inline_math": True,
            "formula_mode": "inline",
            "mixed_columns_risk": True,
        },
    )

    assert _is_unsafe_inline_math_crop_geometry(inline_math_block, "inline_math", page_width=612) is True


def test_narrow_inline_math_crop_is_still_allowed_for_context():
    inline_math_block = block(
        "Comme $n^2$ domine $n$, le terme principal est clair.",
        70,
        120,
        330,
        142,
        metadata={"contains_inline_math": True, "formula_mode": "inline"},
    )

    assert _is_unsafe_inline_math_crop_geometry(inline_math_block, "inline_math", page_width=612) is False
