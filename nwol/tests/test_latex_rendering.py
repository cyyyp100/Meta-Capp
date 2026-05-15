import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.latex import render_formula


def test_truncated_latex_commands_are_ignored_before_mathtext(caplog):
    caplog.set_level(logging.WARNING, logger="LaTeX")

    assert render_formula(r"\tex") is None
    assert render_formula(r"\text") is None

    assert "Rendu LaTeX échoué" not in caplog.text
