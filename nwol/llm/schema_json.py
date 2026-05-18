# llm/schema_json.py — Validation stricte des JSON LLM
from __future__ import annotations

import ast
import json
import logging
import re
import string
import unicodedata
from typing import Any

from core.math_text import repair_common_inline_math_artifacts
from i18n import t
from metacog.reflection import normalize_meta_cognition_questions
from utils.flashcard_tags import normalize_flashcard_tags

logger = logging.getLogger("LLM.schema")

CRITERIA = (
    "attention",
    "context_comprehension",
    "creativity",
    "retention",
    "curiosity",
    "meta_cognition",
)

QUESTION_TYPES = (
    "qcm",
    "open",
    "comprehension",
    "application",
    "curiosity",
    "visualization",
    "metacognition",
    "anticipation",
)

_QUESTION_TYPE_ALIASES = {
    "mcq": "qcm",
    "multiple_choice": "qcm",
    "qcm_verification_rapide_de_comprehension": "qcm",
    "question_ouverte": "open",
    "ouverte": "open",
    "open_question": "open",
    "reformulation": "open",
    "question_de_comprehension": "comprehension",
    "question_de_comprehension_textuelle": "comprehension",
    "comprehension_textuelle": "comprehension",
    "textual_comprehension": "comprehension",
    "question_d_application": "application",
    "question_application": "application",
    "application_question": "application",
    "mise_en_pratique": "application",
    "question_de_curiosite": "curiosity",
    "question_de_curiosite_inductive": "curiosity",
    "curiosite": "curiosity",
    "curiosite_inductive": "curiosity",
    "inductive": "curiosity",
    "question_inductive": "curiosity",
    "visualisation": "visualization",
    "exercice_de_visualisation": "visualization",
    "visualization_exercise": "visualization",
    "question_metacognitive": "metacognition",
    "metacognitive": "metacognition",
    "metacognitive_question": "metacognition",
    "anticipation_auto_evaluation": "anticipation",
    "auto_evaluation": "anticipation",
    "self_evaluation": "anticipation",
    "question_d_anticipation": "anticipation",
}

_AMBIGUOUS_JSON_LATEX_ESCAPE_RE = re.compile(
    r"\\(?:"
    r"bar|begin|beta|big|binom|bmatrix|boldsymbol|"
    r"forall|frac|"
    r"nabla|neg|ne|neq|ngeq|nleq|not|notin|nsim|nu|"
    r"rangle|rightarrow|right|"
    r"tan|tau|text|theta|therefore|times|to|top"
    r")\b"
)

# Fragments caractéristiques des questions génériques émises par le LLM
# quand il n'a pas de contexte suffisant pour générer une question ancrée.
_GENERIC_QUESTION_FRAGMENTS: tuple[str, ...] = (
    "la relation ou les données du passage",
    "les données du passage à un cas",
    "du passage à un cas simple",
    "appliquerais-tu la relation",
)


def _is_generic_question(text: str) -> bool:
    t = text.lower()
    return any(frag in t for frag in _GENERIC_QUESTION_FRAGMENTS)


def parse_question(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), None)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("question"), dict):
        data = data["question"]
    elif isinstance(data.get("questions"), list):
        first_question = next((item for item in data["questions"] if isinstance(item, dict)), None)
        if first_question is not None:
            data = first_question

    question_type = _normalize_question_type(data.get("question_type"))
    question = _coerce_text(data.get("question", data.get("prompt")))
    choices = _coerce_str_list(
        data.get("choices", data.get("options", data.get("propositions")))
    )
    expected_answer = _coerce_text(
        data.get("expected_answer", data.get("expectedAnswer", data.get("answer")))
    )
    session_hint = _coerce_text(
        data.get(
            "session_hint",
            data.get("adaptive_hint", data.get("pause_suggestion", "")),
        )
    )
    source_block_id = _coerce_text(
        data.get("source_block_id", data.get("source_id", data.get("block_id", "")))
    )
    paragraph_mask = _parse_paragraph_mask(data.get("paragraph_mask"))

    evaluation_criteria = _coerce_str_list(
        data.get("evaluation_criteria", data.get("criteria", data.get("criteres", [])))
    )

    choices = [choice.strip() for choice in choices if choice.strip()]

    if question_type not in QUESTION_TYPES:
        question_type = "qcm" if len(choices) >= 3 else "open"
    if not _non_empty_str(question) or not _non_empty_str(expected_answer):
        return None
    if not evaluation_criteria:
        evaluation_criteria = [t("qa.criteria_faithful")]
    # gemma4 envoie des choices même pour les questions non-QCM → on normalise
    if question_type != "qcm":
        choices = []
    if question_type == "qcm":
        if len(choices) < 3:
            return None
        choices = choices[:4]  # tronquer si > 4
    if paragraph_mask is None:
        return None

    question_clean = repair_common_inline_math_artifacts(question.strip())
    if _is_generic_question(question_clean):
        logger.debug("Question générique rejetée : %.120s", question_clean)
        return None

    return {
        "question_type": question_type,
        "question": question_clean,
        "choices": [repair_common_inline_math_artifacts(choice.strip()) for choice in choices],
        "expected_answer": repair_common_inline_math_artifacts(expected_answer.strip()),
        "evaluation_criteria": [
            repair_common_inline_math_artifacts(item.strip())
            for item in evaluation_criteria
            if item.strip()
        ],
        "session_hint": session_hint.strip() if isinstance(session_hint, str) else "",
        "source_block_id": source_block_id.strip() if isinstance(source_block_id, str) else "",
        "paragraph_mask": paragraph_mask,
    }


def _normalize_question_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = _normalize_question_type_token(value)
    if token in QUESTION_TYPES:
        return token
    return _QUESTION_TYPE_ALIASES.get(token)


def _normalize_question_type_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", "_", without_accents.lower()).strip("_")


def parse_evaluation(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    for key in ("evaluation", "answer_evaluation", "answerEvaluation", "result"):
        if isinstance(data.get(key), dict):
            data = data[key]
            break

    verdict = _normalize_verdict(
        data.get("verdict", data.get("grade", data.get("status", data.get("result"))))
    )
    if verdict is None:
        verdict = _verdict_from_score(data.get("score", data.get("completion_score")))
    if verdict is None and isinstance(data.get("is_correct"), bool):
        verdict = "correct" if data["is_correct"] else "incorrect"
    if verdict is None:
        return None
    feedback = _coerce_text(
        data.get("feedback", data.get("comment", data.get("explanation", "")))
    )
    completion_raw = data.get("completion", "")
    hint_raw = data.get("hint", "")
    completion_value = _coerce_text(completion_raw)
    hint_value = _coerce_text(hint_raw)
    if feedback is None or completion_value is None or hint_value is None:
        return None
    if not feedback.strip():
        feedback = {
            "correct": "Réponse acceptée.",
            "partial": "Réponse partielle : il manque une précision.",
            "incorrect": "Réponse insuffisante pour valider ce point.",
        }[verdict]

    signals = data.get("metacog_signals")
    analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
    if not isinstance(signals, dict) and analysis:
        signals = {
            "attention": analysis.get("attentionDelta", analysis.get("attention_delta", 0.0)),
            "curiosity": analysis.get("curiosityDelta", analysis.get("curiosity_delta", 0.0)),
            "creativity": analysis.get("creativityDelta", analysis.get("creativity_delta", 0.0)),
            "context_comprehension": 0.0,
            "retention": 0.0,
            "meta_cognition": 0.0,
        }
    if not isinstance(signals, dict):
        signals = {}
    normalized_signals = {}
    for criterion in CRITERIA:
        value = _number_value(signals.get(criterion, 0.0))
        normalized_signals[criterion] = _clamp(float(value if value is not None else 0.0), -2.0, 2.0)

    flashcard = data.get("flashcard")
    if flashcard in (None, False, "", {}):
        flashcard = None
    else:
        flashcard = parse_flashcard(flashcard)

    curiosity_signals = _parse_curiosity_signals(
        data.get("curiosity_signals")
        or data.get("curiositySignals")
        or analysis.get("curiositySignals")
        or analysis.get("curiosity_signals")
        or {}
    )
    creativity_signals = _parse_creativity_signals(
        data.get("creativity_signals")
        or data.get("creativitySignals")
        or analysis.get("creativitySignals")
        or analysis.get("creativity_signals")
        or {}
    )
    answer_to_user_question = data.get("answer_to_user_question", data.get("answerToUserQuestion"))
    answer_to_user_question = _coerce_text(answer_to_user_question)
    completion = completion_value.strip() if verdict == "partial" else ""
    hint = hint_value.strip() if verdict == "incorrect" else ""

    return {
        "verdict": verdict,
        "feedback": feedback.strip(),
        "completion": completion,
        "hint": hint,
        "metacog_signals": normalized_signals,
        "curiosity_signals": curiosity_signals,
        "creativity_signals": creativity_signals,
        "answer_to_user_question": answer_to_user_question.strip() if isinstance(answer_to_user_question, str) and answer_to_user_question.strip() else None,
        "flashcard": flashcard,
    }


def parse_follow_up(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None

    answer = _coerce_text(data.get("answer", data.get("response", data.get("réponse"))))
    if not _non_empty_str(answer):
        return None

    signals = data.get("metacog_signals")
    if not isinstance(signals, dict):
        signals = {}
    normalized_signals = {}
    for criterion in CRITERIA:
        value = _number_value(signals.get(criterion, 0.0))
        normalized_signals[criterion] = _clamp(float(value if value is not None else 0.0), -2.0, 2.0)
    normalized_signals["curiosity"] = max(1.0, normalized_signals["curiosity"])
    normalized_signals["meta_cognition"] = 0.0

    curiosity_signals = _parse_curiosity_signals(
        data.get("curiosity_signals")
        or data.get("curiositySignals")
        or {}
    )
    curiosity_signals["asked_follow_up_question"] = True

    return {
        "answer": answer.strip(),
        "metacog_signals": normalized_signals,
        "curiosity_signals": curiosity_signals,
    }


def parse_rephrasing(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    rephrased = data.get("rephrased_paragraph") or data.get("reformulation") or data.get("text") or ""
    if not _non_empty_str(rephrased):
        return None
    return {
        "rephrasing_angle": (data.get("rephrasing_angle") or data.get("angle") or "").strip(),
        "rephrased_paragraph": rephrased.strip(),
        "note": (data.get("note") or "").strip(),
    }


def parse_session_summary(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    summary = data.get("session_summary")
    if not isinstance(summary, dict):
        summary = data

    int_fields = ("duration_s", "paragraphs_read", "flashcards_created", "rephrasings_count")
    parsed_ints: dict[str, int] = {}
    for field in int_fields:
        value = _int_value(summary.get(field, 0))
        if value is None or value < 0:
            return None
        parsed_ints[field] = value

    success_rate = _number_value(summary.get("success_rate", summary.get("successRate", 0.0)))
    if success_rate is None:
        return None
    if success_rate > 1.0:
        success_rate = success_rate / 100.0
    qualitative_summary = _coerce_text(
        summary.get("qualitative_summary", summary.get("summary", summary.get("overview")))
    )
    if not _non_empty_str(qualitative_summary):
        return None
    questions = _coerce_str_list(
        summary.get("metacognitive_questions", summary.get("questions", []))
    )
    questions = normalize_meta_cognition_questions(questions)
    if len(questions) != 3:
        return None

    return {
        "session_summary": {
            "duration_s": parsed_ints["duration_s"],
            "paragraphs_read": parsed_ints["paragraphs_read"],
            "flashcards_created": parsed_ints["flashcards_created"],
            "rephrasings_count": parsed_ints["rephrasings_count"],
            "success_rate": _clamp(float(success_rate), 0.0, 1.0),
            "qualitative_summary": qualitative_summary.strip(),
            "metacognitive_questions": questions,
        }
    }


def parse_meta_cognition_questions(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if isinstance(data, list):
        questions = _coerce_str_list(data)
        normalized = normalize_meta_cognition_questions(questions)
        return {"questions": normalized} if len(normalized) == 3 else None
    if not isinstance(data, dict):
        return None
    questions = _coerce_str_list(
        data.get("questions", data.get("metacognitive_questions", data.get("metacognition_questions", [])))
    )
    normalized = normalize_meta_cognition_questions(questions)
    if len(normalized) != 3:
        return None
    return {"questions": normalized}


def parse_meta_cognition_analysis(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("analysis"), dict):
        data = data["analysis"]

    raw_delta = data.get("score_delta", data.get("scoreDelta"))
    delta = _number_value(raw_delta)
    if delta is None:
        delta = 0.0
    raw_score = data.get("score", 50.0)
    score = _number_value(raw_score)
    if score is None:
        score = 50.0
    reasoning = _coerce_text(data.get("reasoning", data.get("rationale", "")))
    if reasoning is None:
        reasoning = ""

    signals = data.get("detected_signals", data.get("detectedSignals"))
    if not isinstance(signals, dict):
        signals = {}
    parsed_signals = {
        "awareness_of_difficulties": _clamp(float(_number_value(_signal_value(signals, "awareness_of_difficulties", "awarenessOfDifficulties")) or 0.0), 0.0, 1.0),
        "strategy_identification": _clamp(float(_number_value(_signal_value(signals, "strategy_identification", "strategyIdentification")) or 0.0), 0.0, 1.0),
        "self_evaluation": _clamp(float(_number_value(_signal_value(signals, "self_evaluation", "selfEvaluation")) or 0.0), 0.0, 1.0),
        "specificity": _clamp(float(_number_value(_signal_value(signals, "specificity")) or 0.0), 0.0, 1.0),
        "honesty_or_depth": _clamp(float(_number_value(_signal_value(signals, "honesty_or_depth", "honestyOrDepth")) or 0.0), 0.0, 1.0),
    }

    return {
        "score_delta": _clamp(float(delta), -20.0, 20.0),
        "score": _clamp(float(score), 0.0, 100.0),
        "reasoning": reasoning.strip(),
        "detected_signals": parsed_signals,
    }


def parse_chapter_summary(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    summary = data.get("chapter_summary")
    if not isinstance(summary, dict):
        summary = data

    title = _coerce_text(summary.get("title", ""))
    overview = _coerce_text(summary.get("overview", summary.get("summary", "")))
    recap = summary.get("recap_qa", summary.get("recap", summary.get("qa", [])))
    if not isinstance(title, str) or not _non_empty_str(overview):
        return None
    if not isinstance(recap, list):
        return None

    parsed_recap = []
    for item in recap[:3]:
        if not isinstance(item, dict):
            continue
        question = _coerce_text(item.get("question", item.get("q")))
        answer = _coerce_text(item.get("answer", item.get("a")))
        if not _non_empty_str(question) or not _non_empty_str(answer):
            continue
        parsed_recap.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    while len(parsed_recap) < 3:
        idx = len(parsed_recap) + 1
        parsed_recap.append({
            "question": f"Quel point clé retenir ({idx}) ?",
            "answer": overview.strip(),
        })

    return {
        "chapter_summary": {
            "title": title.strip(),
            "overview": overview.strip(),
            "recap_qa": parsed_recap,
        }
    }


def parse_curiosity_hook(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    curiosity_hook = _coerce_text(
        data.get("curiosity_hook", data.get("hook", data.get("message")))
    )
    if not _non_empty_str(curiosity_hook):
        return None
    tone = _normalize_curiosity_tone(data.get("tone"))
    if tone not in {"calm", "intriguing", "concrete", "playful"}:
        tone = "concrete"
    link_with_chapter = _coerce_text(
        data.get("link_with_chapter", data.get("linkWithChapter", ""))
    )
    if link_with_chapter is None:
        link_with_chapter = ""
    accessibility = _number_value(
        data.get("estimated_accessibility", data.get("estimatedAccessibility", 0.6))
    )
    if accessibility is None:
        accessibility = 0.6
    if accessibility > 1.0:
        accessibility = accessibility / 100.0
    return {
        "curiosity_hook": curiosity_hook.strip(),
        "tone": tone,
        "link_with_chapter": link_with_chapter.strip(),
        "estimated_accessibility": _clamp(float(accessibility), 0.0, 1.0),
    }


def parse_quiz_session_analysis(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None

    analysis = data.get("analysis", "")
    if not isinstance(analysis, str):
        analysis = ""

    weak_subjects = data.get("weak_subjects", [])
    if not isinstance(weak_subjects, list):
        weak_subjects = []
    weak_subjects = [str(s).strip() for s in weak_subjects if s]

    courses_raw = data.get("courses_to_review", [])
    if not isinstance(courses_raw, list):
        courses_raw = []

    courses = []
    for item in courses_raw[:3]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("subject") or "").strip()
        subject = str(item.get("subject") or "").strip()
        reason = str(item.get("reason") or "").strip()
        document = str(item.get("document") or "").strip()
        chapter_title = str(item.get("chapter_title") or "").strip()
        if title or subject:
            courses.append({
                "title": title or subject,
                "subject": subject,
                "reason": reason,
                "document": document,
                "chapter_title": chapter_title,
            })

    return {
        "analysis": analysis.strip(),
        "weak_subjects": weak_subjects,
        "courses_to_review": courses,
    }


def parse_latex_paragraph_render(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    rendered = _coerce_text(data.get("rendered", data.get("text", data.get("paragraph"))))
    if not _non_empty_str(rendered) or len(rendered.strip()) < 8:
        return None
    return {"rendered": rendered.strip()}


def parse_flashcard_tags(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if isinstance(data, list):
        tags = _coerce_str_list(data)
        normalized = normalize_flashcard_tags(tags)
        if not (2 <= len(normalized) <= 6):
            return None
        return {"tags": normalized[:6]}
    if not isinstance(data, dict):
        return None
    tags = _coerce_str_list(data.get("tags", data.get("labels", data.get("keywords", []))))
    normalized = normalize_flashcard_tags(tags)
    if not (2 <= len(normalized) <= 6):
        return None
    return {"tags": normalized[:6]}


def parse_flashcard(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return None
    front = _coerce_text(data.get("front", data.get("recto")))
    back = _coerce_text(data.get("back", data.get("verso")))
    if not _non_empty_str(front) or not _non_empty_str(back):
        return None
    tags = _coerce_str_list(data.get("tags", []))
    difficulty = _int_value(data.get("difficulty", 2))
    if difficulty not in (1, 2, 3):
        return None
    return {
        "front": front.strip(),
        "back": back.strip(),
        "tags": normalize_flashcard_tags(tags),
        "difficulty": difficulty,
    }


def _parse_curiosity_signals(raw: dict) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raw = {}
    return {
        "asked_follow_up_question": _bool_value(raw, "asked_follow_up_question", "askedFollowUpQuestion"),
        "asked_for_clarification": _bool_value(raw, "asked_for_clarification", "askedForClarification"),
        "asked_for_example": _bool_value(raw, "asked_for_example", "askedForExample"),
        "explored_beyond_required_answer": _bool_value(raw, "explored_beyond_required_answer", "exploredBeyondRequiredAnswer"),
    }


def _parse_creativity_signals(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    depth = _signal_value(raw, "depth_of_reflection", "depthOfReflection")
    if not isinstance(depth, (int, float)):
        depth = 0.0
    return {
        "goes_beyond_prompt": _bool_value(raw, "goes_beyond_prompt", "goesBeyondPrompt"),
        "makes_connections": _bool_value(raw, "makes_connections", "makesConnections"),
        "uses_analogy": _bool_value(raw, "uses_analogy", "usesAnalogy"),
        "personal_reformulation": _bool_value(raw, "personal_reformulation", "personalReformulation"),
        "original_hypothesis": _bool_value(raw, "original_hypothesis", "originalHypothesis"),
        "depth_of_reflection": _clamp(float(depth), 0.0, 1.0),
    }


def _bool_value(raw: dict, *keys: str) -> bool:
    value = _signal_value(raw, *keys)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "oui"}
    return bool(value)


def _signal_value(raw: dict, *keys: str):
    for key in keys:
        if key in raw:
            return raw[key]
    return 0.0


def _parse_paragraph_mask(value) -> dict | None:
    if value is None:
        return {"enabled": False}
    if not isinstance(value, dict):
        return None

    enabled = value.get("enabled", False)
    if isinstance(enabled, int) and not isinstance(enabled, bool):
        enabled = bool(enabled)
    elif isinstance(enabled, str):
        enabled = enabled.lower() in ("true", "1", "yes")
    elif not isinstance(enabled, bool):
        enabled = False  # fallback safe
    if not enabled:
        return {"enabled": False}

    start_char = value.get("start_char")
    end_char = value.get("end_char")
    if not isinstance(start_char, int) or not isinstance(end_char, int):
        return None
    if start_char < 0 or end_char <= start_char:
        return None

    placeholder = value.get("placeholder", t("qa.mask_placeholder"))
    if not _non_empty_str(placeholder):
        return None

    return {
        "enabled": True,
        "start_char": start_char,
        "end_char": end_char,
        "placeholder": placeholder.strip(),
    }


def _load_json(raw: str | dict) -> Any:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    text = _strip_markdown_fence(raw.strip())
    for candidate in _json_candidates(text):
        for variant in _json_parse_variants(candidate):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                pass
            parsed = _load_python_literal(variant)
            if parsed is not None:
                return parsed
    logger.debug("JSON LLM invalide: %s", raw[:200])
    return None


def _strip_markdown_fence(text: str) -> str:
    fenced = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return "\n".join(block.strip() for block in fenced if block.strip()).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    extracted = _extract_json_object(text)
    if extracted is not None and extracted != text:
        candidates.append(extracted)
    candidates.extend(_extract_balanced_json_values(text))
    return _dedupe_strings(candidates)


def _json_parse_variants(text: str) -> list[str]:
    repaired = _escape_invalid_json_backslashes(text)
    roots = [repaired, text] if repaired != text and _AMBIGUOUS_JSON_LATEX_ESCAPE_RE.search(text) else [text]
    if repaired != text and repaired not in roots:
        roots.append(repaired)

    variants: list[str] = []
    for root in roots:
        variants.append(root)
        no_trailing = _remove_trailing_json_commas(root)
        variants.append(no_trailing)
        normalized_literals = _normalize_json_literals_outside_strings(no_trailing)
        variants.append(normalized_literals)
        variants.append(_quote_unquoted_json_keys(normalized_literals))
        variants.append(_complete_truncated_json(no_trailing))
    return _dedupe_strings(variants)


def _load_python_literal(text: str) -> Any:
    variants = [
        text,
        _python_literals_outside_strings(text),
        _quote_unquoted_json_keys(_python_literals_outside_strings(text)),
    ]
    for variant in _dedupe_strings(variants):
        try:
            return ast.literal_eval(variant)
        except (SyntaxError, ValueError, TypeError):
            continue
    return None


def _extract_balanced_json_values(text: str) -> list[str]:
    values: list[str] = []
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        stack: list[str] = []
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                stack.append("}")
            elif current == "[":
                stack.append("]")
            elif current in "}]":
                if not stack or stack[-1] != current:
                    break
                stack.pop()
                if not stack:
                    values.append(text[start:index + 1])
                    break
    return values


def _remove_trailing_json_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _quote_unquoted_json_keys(text: str) -> str:
    return re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        lambda match: f'{match.group(1)}"{match.group(2)}":',
        text,
    )


def _complete_truncated_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()

    suffix = ""
    if in_string:
        if escaped:
            suffix += "\\"
        suffix += '"'
    while stack:
        suffix += stack.pop()
    return text + suffix


def _normalize_json_literals_outside_strings(text: str) -> str:
    return _replace_literals_outside_strings(
        text,
        {"True": "true", "False": "false", "None": "null"},
    )


def _python_literals_outside_strings(text: str) -> str:
    return _replace_literals_outside_strings(
        text,
        {"true": "True", "false": "False", "null": "None"},
    )


def _replace_literals_outside_strings(text: str, replacements: dict[str, str]) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    keys = tuple(sorted(replacements, key=len, reverse=True))
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        replaced = False
        for key in keys:
            if text.startswith(key, index) and _literal_boundary(text, index, len(key)):
                result.append(replacements[key])
                index += len(key)
                replaced = True
                break
        if not replaced:
            result.append(char)
            index += 1
    return "".join(result)


def _literal_boundary(text: str, start: int, length: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[start + length] if start + length < len(text) else ""
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _escape_invalid_json_backslashes(text: str) -> str:
    r"""Repair common LLM JSON errors such as LaTeX ``\sim`` inside strings."""
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    hex_digits = set(string.hexdigits)
    repaired: list[str] = []
    in_string = False
    pending_backslash = False
    i = 0
    while i < len(text):
        char = text[i]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if pending_backslash:
            next_char = text[i + 1] if i + 1 < len(text) else ""
            is_valid_unicode_escape = char == "u" and len(text[i + 1 : i + 5]) == 4 and all(
                item in hex_digits for item in text[i + 1 : i + 5]
            )
            looks_like_latex_command = (
                char.isalpha()
                and not is_valid_unicode_escape
                and (char not in valid_escapes or next_char.isalpha())
            )
            if char in valid_escapes and not looks_like_latex_command and (char != "u" or is_valid_unicode_escape):
                repaired.append("\\")
                repaired.append(char)
            else:
                repaired.append("\\\\")
                repaired.append(char)
            pending_backslash = False
            i += 1
            continue

        if char == "\\":
            pending_backslash = True
            i += 1
            continue

        if char == '"':
            in_string = False
            repaired.append(char)
            i += 1
            continue

        if char == "\n":
            repaired.append("\\n")
        elif char == "\r":
            repaired.append("\\r")
        elif char == "\t":
            repaired.append("\\t")
        else:
            repaired.append(char)
        i += 1

    if pending_backslash:
        repaired.append("\\\\")
    return "".join(repaired)


def _normalize_verdict(value) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[\s_-]+", " ", value.strip().lower())
    aliases = {
        "correct": "correct",
        "correcte": "correct",
        "correct answer": "correct",
        "bonne réponse": "correct",
        "bonne reponse": "correct",
        "réponse correcte": "correct",
        "reponse correcte": "correct",
        "juste": "correct",
        "valid": "correct",
        "valide": "correct",
        "partial": "partial",
        "partiel": "partial",
        "partielle": "partial",
        "partiellement correct": "partial",
        "partiellement correcte": "partial",
        "partly correct": "partial",
        "incomplete": "partial",
        "incomplet": "partial",
        "incomplète": "partial",
        "incomplete answer": "partial",
        "à compléter": "partial",
        "a completer": "partial",
        "incorrect": "incorrect",
        "incorrecte": "incorrect",
        "réponse incorrecte": "incorrect",
        "reponse incorrecte": "incorrect",
        "faux": "incorrect",
        "fausse": "incorrect",
        "wrong": "incorrect",
        "hors sujet": "incorrect",
    }
    return aliases.get(normalized)


def _coerce_text(value) -> str | None:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(item.strip() for item in value if item.strip())
    return None


def _coerce_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, (int, float, bool)):
                result.append(str(item))
            elif isinstance(item, dict):
                text = _coerce_text(
                    item.get("text", item.get("label", item.get("answer", item.get("value"))))
                )
                if text:
                    result.append(text)
        return result
    if isinstance(value, dict):
        return [
            str(item).strip()
            for _key, item in sorted(value.items())
            if isinstance(item, (str, int, float, bool)) and str(item).strip()
        ]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parts = re.split(r"\n+|(?:^|\s)[-•]\s+|;\s*", stripped)
        return [part.strip(" -•") for part in parts if part.strip(" -•")]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    return []


def _verdict_from_score(value) -> str | None:
    score = _number_value(value)
    if score is None:
        return None
    if score > 1.0:
        score = score / 100.0
    if score >= 0.75:
        return "correct"
    if score >= 0.30:
        return "partial"
    return "incorrect"


def _number_value(value) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _int_value(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    number = _number_value(value)
    if number is None:
        return None
    return int(number)


def _non_empty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_fields(data: dict, fields: tuple[str, ...]) -> bool:
    return all(isinstance(data.get(field, ""), str) for field in fields)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_KNOWN_SUBJECTS: frozenset[str] = frozenset({
    "mathématiques", "sciences", "histoire",
    "géographie", "français", "informatique", "culture",
})

_SUBJECT_ALIASES: dict[str, str] = {
    "math": "mathématiques",
    "maths": "mathématiques",
    "mathematics": "mathématiques",
    "mathematiques": "mathématiques",
    "mathématiques": "mathématiques",
    "science": "sciences",
    "sciences": "sciences",
    "history": "histoire",
    "histoire": "histoire",
    "geography": "géographie",
    "geographie": "géographie",
    "géographie": "géographie",
    "french": "français",
    "francais": "français",
    "français": "français",
    "informatique": "informatique",
    "informatics": "informatique",
    "computing": "informatique",
    "computer_science": "informatique",
    "culture": "culture",
    "general": "culture",
    "général": "culture",
    "generale": "culture",
}


def parse_subject_detection(raw: str | dict) -> dict | None:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return {"subject": "culture"}
    subject = _normalize_subject_token(data.get("subject", data.get("matiere", data.get("matière", ""))))
    subject = _SUBJECT_ALIASES.get(subject, subject)
    if subject in _KNOWN_SUBJECTS:
        return {"subject": subject}
    return {"subject": "culture"}


def _normalize_subject_token(value: Any) -> str:
    text = str(value or "").lower().strip()
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", "_", without_accents).strip("_")


def _normalize_curiosity_tone(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = _normalize_subject_token(value)
    aliases = {
        "calme": "calm",
        "calm": "calm",
        "intriguant": "intriguing",
        "intrigant": "intriguing",
        "intriguing": "intriguing",
        "concret": "concrete",
        "concrete": "concrete",
        "ludique": "playful",
        "playful": "playful",
    }
    return aliases.get(token)
