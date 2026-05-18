import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.ollama_client import (
    _fallback_evaluation_from_prompt,
    _fallback_question_from_prompt,
    _generate_json,
    _sanitize_math_paragraph_render,
    _split_paragraph_for_llm,
    _split_paragraph_for_llm_with_context,
    render_math_paragraph_stream_async,
)
from llm.schema_json import parse_evaluation
from llm.schema_json import parse_flashcard_tags, parse_rephrasing
from i18n import set_lang
from llm.prompts import (
    build_evaluation_prompt,
    build_flashcard_tags_prompt,
    build_latex_paragraph_render_text_prompt,
    build_question_prompt,
    build_rephrasing_prompt,
)


def test_latex_render_text_prompt_is_not_json_contract():
    prompt = build_latex_paragraph_render_text_prompt("On a u_n sim n.")

    assert "Réponds uniquement avec le paragraphe corrigé en texte brut" in prompt
    assert '"rendered"' not in prompt


def test_split_paragraph_for_llm_does_not_cut_inside_inline_math():
    formula = "$" + " + ".join(f"x_{index}.y_{index}" for index in range(30)) + "$"
    text = f"Avant. {formula} Après la formule, le paragraphe continue normalement."

    chunks = _split_paragraph_for_llm(text, max_chars=60)

    assert len(chunks) >= 3
    assert all(chunk.count("$") != 1 for chunk in chunks)
    assert any(chunk.startswith("$") and chunk.endswith("$") for chunk in chunks)


def test_split_paragraph_for_llm_contextual_chunks_keep_target_separate():
    text = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80

    chunks = _split_paragraph_for_llm_with_context(
        text,
        max_chars=90,
        previous_context_chars=10,
        next_context_chars=10,
    )

    assert chunks[1]["target"] == "B" * 80
    assert chunks[1]["previous_context"] == "A" * 10
    assert chunks[1]["next_context"] == "C" * 10


def test_math_render_sanitizer_rejects_raw_latex_outside_math():
    source = "A straightforward approach to design h(·,Dmeta;θ) is inspired by SNAIL."
    rendered = r"A straightforward approach to design h(\cdot,Dmeta; \theta) is inspired by SNAIL."

    assert _sanitize_math_paragraph_render(rendered, source) == source


def test_math_render_sanitizer_accepts_delimited_latex():
    source = "On a u_n sim n."
    rendered = r"On a $u_n \sim n$."

    assert _sanitize_math_paragraph_render(rendered, source) == rendered


def test_evaluation_fallback_keeps_qa_loop_from_failing():
    prompt = build_evaluation_prompt(
        {"question": "Que dit le paragraphe ?", "expected_answer": "Une définition."},
        "Il définit la suite comme une fonction.",
        "Une suite est une fonction définie sur les entiers naturels.",
    )

    fallback = _fallback_evaluation_from_prompt(prompt)

    assert fallback["verdict"] == "partial"
    assert fallback["feedback"]
    assert fallback["flashcard"] is None


def test_question_fallback_does_not_treat_3d_unet_as_math():
    prompt = build_question_prompt(
        "The 3D U-Net architecture extends U-Net with volumetric convolutions "
        "for biomedical image segmentation."
    )

    fallback = _fallback_question_from_prompt(prompt)

    assert fallback["question_type"] == "open"
    assert "3D U-Net" in fallback["question"]
    assert "Comment appliquerais-tu la relation ou les données" not in fallback["question"]


def test_question_fallback_uses_application_for_real_formula():
    prompt = build_question_prompt("La relation $y = 2x + 1$ permet de calculer une sortie simple.")

    fallback = _fallback_question_from_prompt(prompt)

    assert fallback["question_type"] == "application"
    assert "formule" in fallback["question"]


def test_question_fallback_uses_english_prompt_language():
    set_lang("en")
    try:
        prompt = build_question_prompt(
            '[Figure on this page: "Figure 1. Process overview"]',
            preferred_question_type="visualization",
        )
        fallback = _fallback_question_from_prompt(prompt)
    finally:
        set_lang("fr")

    assert fallback["question_type"] == "visualization"
    assert fallback["question"].startswith("What should you observe")
    assert "Que dois-tu" not in fallback["question"]


def test_generate_json_repairs_invalid_evaluation_response(monkeypatch):
    from llm import ollama_client

    responses = iter([
        '{"verdict":"almost","feedback":"Idée présente mais imprécise."}',
        '{"verdict":"partial","feedback":"Idée présente mais imprécise.","completion":"","hint":"","flashcard":null}',
    ])

    def fake_call_ollama(prompt, model, images=None, options=None):
        return next(responses)

    monkeypatch.setattr(ollama_client, "_call_ollama", fake_call_ollama)

    parsed = _generate_json("evaluation", "prompt initial", parse_evaluation, model="test", retries=1)

    assert parsed["verdict"] == "partial"
    assert parsed["feedback"].startswith("Idée présente")


def test_generate_json_falls_back_on_ollama_error_for_rephrasing(monkeypatch):
    from llm import ollama_client

    def fake_call_ollama(prompt, model, images=None, options=None):
        raise RuntimeError("Ollama indisponible")

    monkeypatch.setattr(ollama_client, "_call_ollama", fake_call_ollama)

    paragraph = "Le cycle PDCA organise une démarche d'amélioration continue."
    prompt = build_rephrasing_prompt(paragraph, attempt_count=2)

    parsed = _generate_json("rephrasing", prompt, parse_rephrasing, model="test", retries=1)

    assert parsed["rephrased_paragraph"] == paragraph
    assert parsed["note"]


def test_generate_json_falls_back_on_ollama_error_for_flashcard_tags(monkeypatch):
    from llm import ollama_client

    def fake_call_ollama(prompt, model, images=None, options=None):
        raise RuntimeError("Ollama indisponible")

    monkeypatch.setattr(ollama_client, "_call_ollama", fake_call_ollama)

    prompt = build_flashcard_tags_prompt(
        "Analyse asymptotique d'une suite",
        "Utiliser les équivalents pour comparer les termes dominants.",
    )

    parsed = _generate_json("flashcard_tags", prompt, parse_flashcard_tags, model="test", retries=1)

    assert len(parsed["tags"]) >= 2


def test_math_streaming_falls_back_to_json_generation(monkeypatch):
    from llm import ollama_client

    def fake_stream_response(prompt, model, images, on_token, options=None, cancel_token=None):
        raise ValueError("Réponse Ollama vide")

    def fake_generate_json(label, prompt, parser, model, retries=1, image_paths=None, options=None):
        return {"rendered": r"On a $u_n \sim n$."}

    monkeypatch.setattr(ollama_client, "_stream_ollama_response", fake_stream_response)
    monkeypatch.setattr(ollama_client, "_generate_json", fake_generate_json)

    tokens = []
    completed = []
    errors = []
    render_math_paragraph_stream_async(
        "On a u_n sim n.",
        [],
        on_token=tokens.append,
        on_complete=completed.append,
        on_error=errors.append,
    )

    deadline = time.monotonic() + 1.0
    while not completed and not errors and time.monotonic() < deadline:
        time.sleep(0.01)

    assert errors == []
    assert tokens == [r"On a $u_n \sim n$."]
    assert completed == [r"On a $u_n \sim n$."]


def test_math_streaming_completion_uses_json_when_stream_degrades_latex(monkeypatch):
    from llm import ollama_client

    source = "A straightforward approach to design h(·,Dmeta;θ) is inspired by SNAIL."
    degraded = r"A straightforward approach to design h(\cdot,Dmeta; \theta) is inspired by SNAIL."
    repaired = r"A straightforward approach to design $h(\cdot,Dmeta; \theta)$ is inspired by SNAIL."

    def fake_stream_response(prompt, model, images, on_token, options=None, cancel_token=None):
        on_token(degraded)
        return degraded

    def fake_generate_json(label, prompt, parser, model, retries=1, image_paths=None, options=None):
        return {"rendered": repaired}

    monkeypatch.setattr(ollama_client, "_stream_ollama_response", fake_stream_response)
    monkeypatch.setattr(ollama_client, "_generate_json", fake_generate_json)

    tokens = []
    completed = []
    errors = []
    render_math_paragraph_stream_async(
        source,
        [],
        on_token=tokens.append,
        on_complete=completed.append,
        on_error=errors.append,
    )

    deadline = time.monotonic() + 1.0
    while not completed and not errors and time.monotonic() < deadline:
        time.sleep(0.01)

    assert errors == []
    assert tokens == [degraded]
    assert completed == [repaired]
