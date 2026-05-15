import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from document.models import BoundingBox, DocumentBlock
from document.postprocess.context_assets import _looks_like_fragmented_math_text


def test_long_prose_numbered_list_is_not_fragmented_math():
    block = DocumentBlock(
        type="paragraph",
        text=(
            "1) cause instability during training, 2) restrict the model's generalization performance, "
            "3) reduce the framework's flexibility, 4) increase computational overhead and 5) require "
            "costly hyperparameter tuning before it can work robustly on a new task. In this paper we "
            "propose MAML + +, an improved variant with robust training and improved generalization."
        ),
        page=2,
        bbox=BoundingBox(108.0, 517.0, 504.0, 613.0),
        metadata={},
    )

    assert _looks_like_fragmented_math_text(block) is False


def test_broken_inline_latex_text_is_fragmented_math():
    block = DocumentBlock(
        type="paragraph",
        text=(
            r"Then, cross-attention^{{} Atten}$(\textbf {Q},\textbf {K}) = "
            r"\mathrm$ {Softmax_{col}}($\bar$ {$\textbf$ {Q}}$\bar$ {$\textbf$ {K}}^{T} / $\tau$)"
        ),
        page=5,
        bbox=BoundingBox(50.0, 520.0, 545.0, 610.0),
        metadata={"formula_mode": "inline", "contains_inline_math": True},
    )

    assert _looks_like_fragmented_math_text(block) is True
