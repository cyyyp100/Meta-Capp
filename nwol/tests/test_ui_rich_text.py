import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.rich_text import MATH_PATTERN
from core.math_text import repair_common_inline_math_artifacts


def _matches(text: str) -> list[tuple[str, bool]]:
    result = []
    for match in MATH_PATTERN.finditer(text):
        latex = next(group for group in match.groups() if group is not None)
        display = match.group(1) is not None or match.group(2) is not None
        result.append((latex, display))
    return result


def test_math_pattern_supports_common_latex_delimiters():
    text = r"Inline $u_n$ puis \(v_n\), display $$x^2$$ et \[\frac{1}{n}\]."

    assert _matches(text) == [
        ("u_n", False),
        ("v_n", False),
        ("x^2", True),
        (r"\frac{1}{n}", True),
    ]


def test_question_math_repair_restores_split_fomaml_formula():
    text = "Question avec (gFOMAM$L = g$k) dans le texte."
    repaired = repair_common_inline_math_artifacts(text)

    assert repaired == "Question avec ($g^{FOMAML}=g^k$) dans le texte."
    assert _matches(repaired) == [("g^{FOMAML}=g^k", False)]
