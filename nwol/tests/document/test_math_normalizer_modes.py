from document.models import BoundingBox, DocumentBlock
from document.postprocess.inline_formula_repair import repair_fragmented_inline_formulas
from document.postprocess.math_normalizer import normalize_math_blocks
from document.postprocess.quality import evaluate_blocks


def block(text, page=1, x0=60, y0=100, x1=500, y1=115, btype="paragraph", metadata=None):
    return DocumentBlock(
        type=btype,
        text=text,
        page=page,
        bbox=BoundingBox(x0, y0, x1, y1),
        metadata=dict(metadata or {}),
    )


def test_inline_mode_prevents_formula_block_conversion():
    result = normalize_math_blocks(
        [
            block(
                "u_n = 1 / n",
                metadata={"formula_mode": "inline", "contains_inline_math": True},
            )
        ]
    )

    assert result[0].type == "paragraph"
    assert result[0].metadata["formula_mode"] == "inline"


def test_ambiguous_mode_prevents_formula_block_conversion():
    result = normalize_math_blocks(
        [
            block(
                "1 / n",
                metadata={"formula_mode": "ambiguous", "contains_inline_math": True},
            )
        ]
    )

    assert result[0].type == "paragraph"
    assert result[0].metadata["formula_mode"] == "ambiguous"


def test_repair_fragmented_inline_formula_merges_short_display_between_paragraphs():
    blocks = [
        block("u_n = (n² − n) ln", y0=100, y1=112),
        block(
            "1 +",
            y0=116,
            y1=128,
            btype="formula",
            metadata={"formula_mode": "display", "render_mode": "pdf_crop"},
        ),
        block("1/n).", y0=132, y1=144),
    ]

    repaired = repair_fragmented_inline_formulas(blocks)
    normalized = normalize_math_blocks(repaired)

    assert len(normalized) == 1
    assert normalized[0].type == "paragraph"
    assert normalized[0].metadata["repaired_fragmented_inline_formula"] is True
    assert "[formule]" not in normalized[0].text


def test_quality_warns_about_fragmented_inline_formula_pattern():
    blocks = [
        block("u_n = (n² − n) ln", y0=100, y1=112),
        block(
            "1 +",
            y0=116,
            y1=128,
            btype="formula",
            metadata={"formula_mode": "display", "render_mode": "pdf_crop"},
        ),
        block("1/n).", y0=132, y1=144),
    ]

    score, warnings = evaluate_blocks(blocks)

    assert score < 1.0
    assert "Certaines formules inline semblent avoir été fragmentées." in warnings


def test_quality_does_not_flag_common_scientific_camel_acronyms_as_glued_text():
    blocks = [
        block("The NIfTI volume is processed with ReLU layers and RoIAlign features."),
        block("CycleGAN, SinGAN-Seg, CoRR, OpenAI, interSD and SwinUNETR’s outputs are normal scientific tokens.", y0=130, y1=145),
    ]

    score, warnings = evaluate_blocks(blocks)

    assert score == 1.0
    assert "Des mots ou symboles semblent encore collés." not in warnings


def test_quality_accepts_fragmented_math_paragraph_when_context_crop_exists():
    risky = block(
        r"The matrix is [\frac{x}{y}] in the extracted text.",
        metadata={"context_asset_path": "/tmp/context.png"},
    )

    score, warnings = evaluate_blocks([risky])

    assert score == 1.0
    assert "Certaines formules LaTeX semblent encore mal reconstruites." not in warnings


def test_quality_accepts_valid_bracketed_inline_expectation():
    blocks = [
        block(r"This corresponds to the expected loss E_{$\tau$} [L_{$\tau$}] during training."),
    ]

    score, warnings = evaluate_blocks(blocks)

    assert score == 1.0
    assert "Certaines formules LaTeX semblent encore mal reconstruites." not in warnings


def test_repair_ambiguous_inline_sequence_keeps_text_instead_of_formula_block():
    blocks = [
        block(
            "u_n = (n² − n) ln",
            y0=100,
            y1=112,
            metadata={"formula_mode": "ambiguous", "contains_inline_math": True},
        ),
        block(
            "1 +",
            x0=280,
            y0=150,
            x1=320,
            y1=162,
            metadata={"formula_mode": "ambiguous", "contains_inline_math": True},
        ),
        block(
            "1/n).",
            x0=280,
            y0=180,
            x1=340,
            y1=192,
            metadata={"formula_mode": "ambiguous", "contains_inline_math": True},
        ),
    ]

    repaired = repair_fragmented_inline_formulas(blocks)
    normalized = normalize_math_blocks(repaired)

    assert len(normalized) == 1
    assert normalized[0].type == "paragraph"
    assert normalized[0].metadata["repaired_ambiguous_inline_sequence"] is True
    assert "[formule]" not in normalized[0].text


def test_repair_inline_result_formula_after_obtient():
    blocks = [
        block(
            "Comme n² − n ∼ n² et ln(1 + 1/n) ∼ 1/n, on obtient u_n ∼ n² · 1",
            x0=95,
            y0=100,
            x1=560,
            y1=114,
            metadata={"formula_mode": "inline", "contains_inline_math": True},
        ),
        block(
            ". 1/n = n.",
            x0=95,
            y0=134,
            x1=190,
            y1=154,
            btype="formula",
            metadata={"formula_mode": "display", "render_mode": "pdf_crop"},
        ),
    ]

    repaired = repair_fragmented_inline_formulas(blocks)
    normalized = normalize_math_blocks(repaired)

    assert len(normalized) == 1
    assert normalized[0].type == "paragraph"
    assert normalized[0].metadata["repaired_inline_result_formula"] is True
    assert "1 / n = n" in normalized[0].text


def test_repair_inline_formula_bridge_with_words_between_paragraph_parts():
    blocks = [
        block(
            "Hence, in Reptile we replace W^{∗}",
            x0=95,
            y0=572,
            x1=520,
            y1=616,
            metadata={"formula_mode": "inline", "contains_inline_math": True},
        ),
        block(
            "\\tau (\\phi) by the",
            x0=240,
            y0=605,
            x1=360,
            y1=619,
            btype="formula",
            metadata={"formula_mode": "display", "render_mode": "pdf_crop"},
        ),
        block("result of running k steps of gradient descent.", x0=95, y0=619, x1=520, y1=664),
    ]

    repaired = repair_fragmented_inline_formulas(blocks)
    normalized = normalize_math_blocks(repaired)

    assert len(normalized) == 1
    assert normalized[0].type == "paragraph"
    assert normalized[0].metadata["repaired_fragmented_inline_formula"] is True
    assert "by the result" in normalized[0].text
