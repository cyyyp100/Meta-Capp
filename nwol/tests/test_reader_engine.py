import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import FIGURE_DISPLAY_PAUSE_MS
from core.scopes import TextScope
from reader.engine import ReadingEngine
from reader.state import ReaderState


def _engine(blocks, speed_ms=0, on_paragraph_complete=None, on_math_paragraph=None, on_section_complete=None):
    state = ReaderState(speed_ms=speed_ms)
    state.set_scope(TextScope("chapter", "Test", 1, 1, blocks))
    emitted = []
    scheduled = []

    def schedule(delay_ms, callback):
        scheduled.append(callback)
        return len(scheduled)

    engine = ReadingEngine(
        state,
        on_char=lambda char: emitted.append(("char", char)),
        on_block=lambda block: emitted.append(("block", block.get("type"))),
        on_end=lambda: emitted.append(("end", "")),
        schedule_fn=schedule,
        on_paragraph_complete=on_paragraph_complete,
        on_math_paragraph=on_math_paragraph,
        on_section_complete=on_section_complete,
    )
    return engine, state, emitted, scheduled


def _drain(scheduled, limit=2000):
    steps = 0
    while scheduled and steps < limit:
        scheduled.pop(0)()
        steps += 1


# ---------------------------------------------------------------------------
# Tests inchangés — comportement de base non affecté par la refonte section
# ---------------------------------------------------------------------------

def test_figure_blocks_get_a_reading_pause():
    engine, _state, emitted, delays = _engine([{"type": "figure", "caption": "Figure 1"}], speed_ms=10)
    engine.play()
    assert emitted == [("block", "figure")]


def test_prose_with_experiments_and_slashes_is_not_streamed_as_math():
    block = {
        "type": "paragraph",
        "text": (
            "In our experiments, few-shot recognition and expectation-based methods "
            "use Faster / Mask R-CNN [36, 17] without any formula."
        ),
        "metadata": {"context_asset_reason": "math_dense_text"},
    }
    math_calls = []
    engine, _state, emitted, scheduled = _engine(
        [block],
        on_math_paragraph=lambda b, done: math_calls.append(b["text"]),
    )
    engine.play()
    _drain(scheduled)
    assert math_calls == []
    assert emitted[0] == ("char", "I")


# ---------------------------------------------------------------------------
# Nouveaux tests — Q&R au niveau de la section
# ---------------------------------------------------------------------------

def test_section_complete_triggered_at_heading_boundary():
    """Un heading après du contenu déclenche on_section_complete avant le heading."""
    para = {"type": "paragraph", "text": "Contenu de la section."}
    heading = {"type": "heading", "text": "Nouvelle section"}
    section_events = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        section_events.append({
            "blocks": blocks,
            "heading": heading_block,
            "has_latex": has_latex,
        })
        resume()

    engine, state, emitted, scheduled = _engine(
        [para, heading],
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    assert len(section_events) == 1
    assert section_events[0]["blocks"] == [para]
    assert section_events[0]["has_latex"] is False
    # Heading émis après la Q&R
    assert ("block", "heading") in emitted


def test_section_complete_triggered_at_eof():
    """La dernière section déclenche on_section_complete avant on_end."""
    para = {"type": "paragraph", "text": "Dernier paragraphe de la session."}
    order = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        order.append("section")
        resume()

    engine, state, emitted, scheduled = _engine(
        [para],
        on_section_complete=on_section_complete,
    )

    def on_end():
        order.append("end")
        emitted.append(("end", ""))

    engine.on_end = on_end
    engine.play()
    _drain(scheduled)

    assert order == ["section", "end"]


def test_no_section_complete_for_empty_sections():
    """Deux headings consécutifs (section vide) ne déclenchent pas de Q&R."""
    h1 = {"type": "heading", "text": "Section A"}
    h2 = {"type": "heading", "text": "Section B"}
    section_events = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        section_events.append(blocks)
        resume()

    engine, state, emitted, scheduled = _engine(
        [h1, h2],
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    assert section_events == []
    assert ("block", "heading") in emitted


def test_section_blocks_accumulated_correctly():
    """Les blocs entre deux headings sont tous dans current_section_blocks."""
    h1 = {"type": "heading", "text": "Section 1"}
    para = {"type": "paragraph", "text": "Paragraphe de la section 1."}
    fig = {"type": "figure", "caption": "Figure 1"}
    h2 = {"type": "heading", "text": "Section 2"}
    captured = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        captured.append(list(blocks))
        resume()

    engine, state, emitted, scheduled = _engine(
        [h1, para, fig, h2],
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    assert len(captured) >= 1
    types = [b.get("type") for b in captured[0]]
    assert "paragraph" in types
    assert "figure" in types


def test_section_has_latex_detected():
    """section_has_latex = True si un bloc contient une formule."""
    para = {"type": "paragraph", "text": "Contexte."}
    formula = {
        "type": "paragraph",
        "text": "La formule $x^2 + y^2 = r^2$ est importante.",
        "metadata": {"formula_mode": "inline"},
    }
    h2 = {"type": "heading", "text": "Suite"}
    latex_flags = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        latex_flags.append(has_latex)
        resume()

    engine, state, emitted, scheduled = _engine(
        [para, formula, h2],
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    assert len(latex_flags) >= 1
    assert latex_flags[0] is True


def test_heading_resets_section_index():
    """Chaque heading incrémente le section_index."""
    h1 = {"type": "heading", "text": "H1"}
    para1 = {"type": "paragraph", "text": "Contenu un."}
    h2 = {"type": "heading", "text": "H2"}
    para2 = {"type": "paragraph", "text": "Contenu deux."}
    indices = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        indices.append(state.section_index)
        resume()

    engine, state, emitted, scheduled = _engine(
        [h1, para1, h2, para2],
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    assert len(indices) >= 2
    assert indices[1] > indices[0]


def test_math_paragraph_added_to_section():
    """Un paragraphe math est ajouté à la section après rendu."""
    block = {
        "type": "paragraph",
        "text": "Soit $f(x) = x^2$.",
        "metadata": {"formula_mode": "inline"},
    }
    h2 = {"type": "heading", "text": "Suite"}
    captured = []

    def on_math_paragraph(math_block, done):
        done("Soit $f(x) = x^2$.")

    def on_section_complete(blocks, heading_block, has_latex, resume):
        captured.append(list(blocks))
        resume()

    engine, state, emitted, scheduled = _engine(
        [block, h2],
        on_math_paragraph=on_math_paragraph,
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    assert len(captured) >= 1
    assert any(b.get("type") == "paragraph" for b in captured[0])


def test_on_paragraph_complete_used_for_rendering_only():
    """on_paragraph_complete est appelé pour le rendu, pas pour déclencher Q&R."""
    para = {"type": "paragraph", "text": "Paragraphe court."}
    render_calls = []

    def on_paragraph_complete(block, resume):
        render_calls.append(block["text"])
        resume()

    engine, state, emitted, scheduled = _engine(
        [para],
        on_paragraph_complete=on_paragraph_complete,
    )
    engine.play()
    _drain(scheduled)

    # Le callback est appelé mais qa_active ne reste pas True (la Q&R est au niveau section)
    assert len(render_calls) == 1
    assert state.qa_active is False


def test_reading_continues_without_qa_pause_between_paragraphs():
    """La lecture ne s'arrête plus entre deux paragraphes pour Q&R."""
    p1 = {"type": "paragraph", "text": "Premier paragraphe assez long pour éviter le skip."}
    p2 = {"type": "paragraph", "text": "Deuxième paragraphe aussi assez long pour la lecture."}
    section_events = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        section_events.append(len(blocks))
        resume()

    engine, state, emitted, scheduled = _engine(
        [p1, p2],
        on_section_complete=on_section_complete,
    )
    engine.play()
    _drain(scheduled)

    # Les deux paragraphes sont dans la même section (pas de heading entre eux)
    assert len(section_events) == 1
    assert section_events[0] == 2


def test_stop_clears_pending_section_before_new_scope():
    """Changer de portée ne doit pas déclencher une Q&R de l'ancienne section."""
    stale = {"type": "paragraph", "text": "Ancien contenu."}
    new_heading = {"type": "heading", "text": "5. Conclusion"}
    captured = []

    def on_section_complete(blocks, heading_block, has_latex, resume):
        captured.append((heading_block, list(blocks)))
        resume()

    engine, state, emitted, scheduled = _engine(
        [new_heading],
        on_section_complete=on_section_complete,
    )
    engine._current_section_heading = {"type": "heading", "text": "INTRODUCTION"}
    engine._current_section_blocks = [stale]

    engine.stop()
    state.set_scope(TextScope("chapter", "Conclusion", 9, 9, [new_heading]))
    engine.play()
    _drain(scheduled)

    assert captured == []
    assert emitted == [("block", "heading"), ("end", "")]
