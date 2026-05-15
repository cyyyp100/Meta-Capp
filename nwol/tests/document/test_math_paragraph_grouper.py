from document.models import BoundingBox, DocumentBlock
from document.postprocess.math_paragraph_grouper import group_math_dense_paragraphs_until_heading


def block(text, y0, y1, btype="paragraph", metadata=None):
    return DocumentBlock(
        type=btype,
        text=text,
        page=1,
        bbox=BoundingBox(60, y0, 520, y1),
        metadata=dict(metadata or {}),
    )


def test_math_dense_paragraph_groups_until_next_subheading():
    blocks = [
        block(
            "Comme n² − n ∼ n² et ln(1 + 1/n) ∼ 1/n, on obtient u_n ∼ n² · 1",
            100,
            114,
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
        block(". 1/n = n.", 140, 162, btype="formula", metadata={"formula_mode": "display"}),
        block("On peut donc comparer les ordres de grandeur.", 170, 184),
        block("1.1.4 Autre méthode", 230, 250, btype="subheading"),
        block("Ce nouveau passage ne doit pas être fusionné.", 260, 274),
    ]

    grouped = group_math_dense_paragraphs_until_heading(blocks)

    assert len(grouped) == 3
    assert grouped[0].type == "paragraph"
    assert grouped[0].metadata["math_dense_group"] is True
    assert "1/n = n" in grouped[0].text
    assert "On peut donc comparer" in grouped[0].text
    assert grouped[1].type == "subheading"


def test_plain_paragraph_does_not_start_group():
    blocks = [
        block("Ce paragraphe contient une explication ordinaire sans calcul dense.", 100, 114),
        block("Une suite simple.", 120, 134),
    ]

    grouped = group_math_dense_paragraphs_until_heading(blocks)

    assert len(grouped) == 2


def test_inline_variables_in_prose_do_not_absorb_following_display_formula():
    blocks = [
        block(
            "Here, Reptile converges towards $\\phi$ near each task $\\tau$ manifold, "
            "and W_{$\\tau$} denotes optimal parameters for task $\\tau$.",
            100,
            126,
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
        block("2 D(\\phi, W_{\\tau})^{2}", 145, 162, btype="formula", metadata={"formula_mode": "display"}),
        block("We will show that Reptile corresponds to SGD on that objective.", 180, 194),
    ]

    grouped = group_math_dense_paragraphs_until_heading(blocks)

    assert [block.type for block in grouped] == ["paragraph", "formula", "paragraph"]


def test_numbered_display_equation_paragraph_does_not_start_math_dense_group():
    blocks = [
        block(
            "$(43) = E$ $\\tau$ [$\\phi$ - $P_{W\\tau}(\\phi)$], where $P_{W\\tau}(\\phi) = \\arg\\min$",
            100,
            126,
            metadata={"contains_inline_math": True, "formula_mode": "inline"},
        ),
        block("D(p, \\phi)", 132, 146, btype="formula", metadata={"formula_mode": "display"}),
        block("Each iteration of Reptile corresponds to sampling a task.", 158, 172),
    ]

    grouped = group_math_dense_paragraphs_until_heading(blocks)

    assert [block.type for block in grouped] == ["paragraph", "formula", "paragraph"]
    assert all(not block.metadata.get("math_dense_group") for block in grouped)
