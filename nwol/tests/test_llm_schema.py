# tests/test_llm_schema.py
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.schema_json import (
    QUESTION_TYPES,
    parse_chapter_summary,
    parse_curiosity_hook,
    parse_evaluation,
    parse_flashcard,
    parse_flashcard_tags,
    parse_follow_up,
    parse_meta_cognition_analysis,
    parse_meta_cognition_questions,
    parse_question,
    parse_rephrasing,
    parse_session_summary,
)
from i18n import set_lang
from llm.prompts import (
    build_evaluation_prompt,
    build_follow_up_prompt,
    build_question_prompt,
    build_schema_render_prompt,
    build_slide_analysis_prompt,
    build_table_render_prompt,
)


def test_parse_question_valid_open():
    raw = {
        "question_type": "open",
        "question": "Pourquoi cette hypothèse est-elle utile ?",
        "choices": [],
        "expected_answer": "Elle permet de simplifier le raisonnement.",
        "evaluation_criteria": ["Mentionne la simplification."],
    }

    parsed = parse_question(raw)
    assert parsed["question_type"] == "open"
    assert parsed["paragraph_mask"]["enabled"] is False
    assert parsed["session_hint"] == ""


def test_parse_question_keeps_session_hint():
    parsed = parse_question({
        "question_type": "metacognition",
        "question": "Comment vérifies-tu ta compréhension ?",
        "choices": [],
        "expected_answer": "L'utilisateur décrit une stratégie.",
        "evaluation_criteria": ["Décrit une stratégie."],
        "session_hint": "Prends une pause courte avant de continuer.",
    })

    assert parsed["session_hint"].startswith("Prends une pause")


def test_parse_question_keeps_source_block_id():
    parsed = parse_question({
        "question_type": "open",
        "question": "Que faut-il retenir ?",
        "choices": [],
        "expected_answer": "L'idée principale.",
        "evaluation_criteria": ["Réponse fidèle."],
        "source_block_id": "p4:b12:abcdef123456",
    })

    assert parsed["source_block_id"] == "p4:b12:abcdef123456"


def test_parse_question_accepts_all_canonical_question_types():
    for question_type in QUESTION_TYPES:
        parsed = parse_question({
            "question_type": question_type,
            "question": f"Question de type {question_type} ?",
            "choices": ["A", "B", "C"] if question_type == "qcm" else [],
            "expected_answer": "Réponse attendue.",
            "evaluation_criteria": ["Critère."],
        })

        assert parsed is not None
        assert parsed["question_type"] == question_type


def test_parse_question_accepts_french_question_type_aliases():
    parsed = parse_question({
        "question_type": "Question de curiosité / inductive",
        "question": "T'es-tu déjà demandé pourquoi cette hypothèse change le résultat ?",
        "choices": [],
        "expected_answer": "La réponse doit formuler une intuition liée au passage.",
        "evaluation_criteria": ["Reste ancré dans le passage."],
    })

    assert parsed["question_type"] == "curiosity"


def test_parse_question_requires_three_qcm_choices():
    assert parse_question({
        "question_type": "QCM",
        "question": "Quel choix correspond au passage ?",
        "choices": ["A", "B"],
        "expected_answer": "A",
        "evaluation_criteria": ["Choisit la bonne proposition."],
    }) is None

    parsed = parse_question({
        "question_type": "QCM",
        "question": "Quel choix correspond au passage ?",
        "choices": ["A", "B", "C"],
        "expected_answer": "A",
        "evaluation_criteria": ["Choisit la bonne proposition."],
    })

    assert parsed["question_type"] == "qcm"
    assert parsed["choices"] == ["A", "B", "C"]


def test_parse_question_with_paragraph_mask():
    raw = {
        "question_type": "open",
        "question": "Quel mot manque dans ce passage ?",
        "choices": [],
        "expected_answer": "Le mot clé.",
        "evaluation_criteria": ["Identifie le mot masqué."],
        "paragraph_mask": {
            "enabled": True,
            "start_char": 4,
            "end_char": 12,
            "placeholder": "réponse masquée temporairement",
        },
    }

    parsed = parse_question(raw)

    assert parsed["paragraph_mask"]["enabled"] is True
    assert parsed["paragraph_mask"]["start_char"] == 4


def test_parse_question_repairs_unescaped_latex_backslashes():
    raw = r'''
    {
      "question_type": "comprehension",
      "question": "En utilisant $n^2 - n \sim n^2$ et $\ln(1+1/n) \sim 1/n$, que vaut $u_n$ ?",
      "choices": [
        "A. $u_n \sim n$",
        "B. $u_n \sim n^2$",
        "C. $u_n \sim n/2$",
        "D. $u_n \sim 0$"
      ],
      "expected_answer": "$u_n \sim n$",
      "evaluation_criteria": [
        "Reconnaît les équivalents asymptotiques",
        "Conserve les commandes \theta et \nabla comme texte LaTeX"
      ]
    }
    '''

    parsed = parse_question(raw)

    assert parsed is not None
    assert parsed["question_type"] == "comprehension"
    assert r"\sim" in parsed["question"]
    assert parsed["choices"] == []
    assert parsed["expected_answer"] == r"$u_n \sim n$"
    assert r"\theta" in parsed["evaluation_criteria"][1]
    assert r"\nabla" in parsed["evaluation_criteria"][1]


def test_parse_question_repairs_valid_json_escape_latex_commands_first():
    raw = r'''
    {
      "question_type": "open",
      "question": "Que représente l'angle $\theta$ dans cette formule ?",
      "choices": [],
      "expected_answer": "L'angle noté $\theta$.",
      "evaluation_criteria": ["Préserve \theta sans le convertir en tabulation."]
    }
    '''

    parsed = parse_question(raw)

    assert parsed is not None
    assert r"\theta" in parsed["question"]
    assert r"\theta" in parsed["expected_answer"]
    assert r"\theta" in parsed["evaluation_criteria"][0]
    assert "\t" not in parsed["question"]


def test_parse_question_repairs_split_fomaml_inline_math():
    parsed = parse_question({
        "question_type": "curiosity",
        "question": (
            "Pourquoi le FOMAML shared-tail dégrade-t-il les performances "
            "lorsque le méta-gradient (gFOMAM$L = g$k) chevauche les mini-batches ?"
        ),
        "choices": [],
        "expected_answer": "Le chevauchement rend $gFOMAM$L = g$k moins informatif.",
        "evaluation_criteria": ["Explique le rôle de gFOMAML = gk."],
    })

    assert parsed is not None
    assert "$g^{FOMAML}=g^k$" in parsed["question"]
    assert "$g^{FOMAML}=g^k$" in parsed["expected_answer"]
    assert "$g^{FOMAML}=g^k$" in parsed["evaluation_criteria"][0]


def test_parse_question_recovers_markdown_trailing_commas_and_unquoted_keys():
    raw = '''
    Le résultat demandé :
    ```json
    {
      question_type: "open",
      question: "Pourquoi cette méthode est-elle utile ?",
      choices: [],
      expected_answer: "Elle structure le raisonnement.",
      evaluation_criteria: ["Explique le rôle de la méthode.",],
    }
    ```
    '''

    parsed = parse_question(raw)

    assert parsed["question_type"] == "open"
    assert parsed["expected_answer"] == "Elle structure le raisonnement."


def test_parse_question_salvages_slightly_truncated_json():
    raw = (
        '{"question_type":"open","question":"Quel est le rôle du cycle PDCA ?",'
        '"choices":[],"expected_answer":"Il sert à organiser l’amélioration continue'
    )

    parsed = parse_question(raw)

    assert parsed["question"].startswith("Quel est le rôle")
    assert "amélioration continue" in parsed["expected_answer"]


def test_parse_question_invalid_missing_field():
    assert parse_question({"question_type": "open", "question": "Q", "choices": []}) is None


def test_parse_evaluation_with_flashcard():
    raw = {
        "verdict": "partial",
        "feedback": "Idée correcte mais incomplète.",
        "completion": "Ajoute la condition initiale.",
        "hint": "",
        "metacog_signals": {
            "context_comprehension": 0.5,
            "creativity": 0.0,
            "attention": 0.1,
            "retention": 0.0,
            "curiosity": 0.2,
            "meta_cognition": 0.0,
        },
        "curiosity_signals": {
            "asked_follow_up_question": True,
            "asked_for_clarification": False,
            "asked_for_example": False,
            "explored_beyond_required_answer": False,
        },
        "creativity_signals": {
            "goes_beyond_prompt": True,
            "makes_connections": True,
            "uses_analogy": False,
            "personal_reformulation": True,
            "original_hypothesis": False,
            "depth_of_reflection": 0.7,
        },
        "answer_to_user_question": "Oui, ce point se relie au paragraphe source.",
        "flashcard": {
            "front": "Condition ?",
            "back": "Condition initiale.",
            "tags": ["analyse"],
            "difficulty": 2,
        },
    }

    parsed = parse_evaluation(raw)

    assert parsed["verdict"] == "partial"
    assert parsed["flashcard"]["tags"] == ["analyse"]
    assert parsed["curiosity_signals"]["asked_follow_up_question"] is True
    assert parsed["answer_to_user_question"].startswith("Oui")


def test_parse_evaluation_keeps_hint_only_for_incorrect_and_completion_only_for_partial():
    parsed = parse_evaluation({
        "verdict": "correct",
        "feedback": "Correct.",
        "completion": "Texte à ne pas afficher.",
        "hint": "Indice à ne pas afficher.",
        "metacog_signals": {
            "context_comprehension": 0.5,
            "creativity": 0.0,
            "attention": 0.0,
            "retention": 0.0,
            "curiosity": 0.0,
            "meta_cognition": 0.0,
        },
        "curiosity_signals": {},
        "creativity_signals": {},
        "answer_to_user_question": None,
        "flashcard": None,
    })

    assert parsed["completion"] == ""
    assert parsed["hint"] == ""


def test_parse_evaluation_tolerates_common_llm_schema_drift():
    parsed = parse_evaluation({
        "verdict": "partiel",
        "feedback": None,
        "completion": "Ajoute la notation utile.",
        "hint": None,
        "metacog_signals": {
            "context_comprehension": "0.5",
            "curiosity": "1",
        },
        "curiositySignals": {"askedFollowUpQuestion": "true"},
        "creativity_signals": {},
        "answer_to_user_question": None,
        "flashcard": {"front": "", "back": "", "tags": [], "difficulty": 9},
    })

    assert parsed["verdict"] == "partial"
    assert parsed["feedback"]
    assert parsed["completion"] == "Ajoute la notation utile."
    assert parsed["flashcard"] is None
    assert parsed["metacog_signals"]["context_comprehension"] == 0.5
    assert parsed["curiosity_signals"]["asked_follow_up_question"] is True


def test_parse_evaluation_accepts_nested_score_and_list_feedback():
    parsed = parse_evaluation({
        "evaluation": {
            "score": 0.6,
            "feedback": ["L'idée est présente.", "Il manque la notation."],
            "completion": "",
            "hint": "",
        }
    })

    assert parsed["verdict"] == "partial"
    assert parsed["feedback"] == "L'idée est présente. Il manque la notation."


def test_evaluation_prompt_does_not_answer_embedded_follow_up_question():
    prompt = build_evaluation_prompt(
        {"expected_answer": "L'idée principale."},
        "L'idée principale. Mais pourquoi ?",
        "Le paragraphe présente l'idée principale.",
    )

    assert "verdict vaut \"correct\" si l'idée attendue est présente" in prompt
    assert "mets answer_to_user_question à null" in prompt
    assert "hint doit être renseigné uniquement si verdict vaut \"incorrect\"" in prompt


def test_parse_follow_up_forces_curiosity_signal():
    parsed = parse_follow_up({
        "answer": "Le paragraphe indique que l'hypothèse sert à simplifier le raisonnement.",
        "metacog_signals": {
            "context_comprehension": 0.2,
            "creativity": 0.0,
            "attention": 0.0,
            "retention": 0.0,
            "curiosity": 0.0,
            "meta_cognition": 1.0,
        },
        "curiosity_signals": {
            "asked_follow_up_question": False,
            "asked_for_clarification": True,
        },
    })

    assert parsed["answer"].startswith("Le paragraphe")
    assert parsed["metacog_signals"]["curiosity"] >= 1.0
    assert parsed["metacog_signals"]["meta_cognition"] == 0.0
    assert parsed["curiosity_signals"]["asked_follow_up_question"] is True


def test_parse_follow_up_accepts_missing_signals():
    parsed = parse_follow_up({
        "answer": "Le passage donne assez d'éléments pour répondre simplement."
    })

    assert parsed["metacog_signals"]["curiosity"] >= 1.0
    assert parsed["curiosity_signals"]["asked_follow_up_question"] is True


def test_question_prompt_reuses_existing_question_when_present():
    prompt = build_question_prompt("Quelle relation observe-t-on ?", has_existing_question=True)

    assert "Le paragraphe contient déjà une ou plusieurs questions" in prompt
    assert "question principale doit être celle du texte d'origine" in prompt


def test_question_prompt_english_enforces_user_facing_language():
    set_lang("en")
    try:
        prompt = build_question_prompt(
            "Figure 1 shows a process with prompts and image embeddings.",
            preferred_question_type="open",
        )
    finally:
        set_lang("fr")

    assert "Write every user-facing string in English" in prompt
    assert "Écris tous les champs visibles" not in prompt


def test_schema_render_prompt_uses_english_ui_language():
    set_lang("en")
    try:
        prompt = build_schema_render_prompt("Figure 1. Process overview.")
        slide_prompt = build_slide_analysis_prompt()
        table_prompt = build_table_render_prompt("Table 1. Results.")
    finally:
        set_lang("fr")

    assert "Text in English" in prompt
    assert "Available caption: Figure 1. Process overview." in prompt
    assert "Texte en français" not in prompt
    assert "Text in English" in slide_prompt
    assert "Table title or caption: Table 1. Results." in table_prompt


def test_question_prompt_exposes_pedagogical_type_catalog():
    prompt = build_question_prompt("Ce passage définit une notion puis donne un exemple.")

    assert "Types de questions disponibles" in prompt
    assert "Ne choisis pas toujours \"comprehension\"" in prompt
    assert "Question d'application" in prompt
    assert "Anticipation / auto-évaluation" in prompt
    for question_type in QUESTION_TYPES:
        assert f'"{question_type}"' in prompt


def test_question_prompt_uses_gauges_for_adaptive_type():
    random.seed(0)
    prompt = build_question_prompt(
        "Ce passage définit une méthode complexe.",
        session_gauges={
            "attention": 40,
            "context_comprehension": 42,
            "curiosity": 50,
            "meta_cognition": 50,
        },
    )

    assert 'Type pédagogique cible : "metacognition"' in prompt
    assert "Attention actuelle sous le seuil 45" in prompt
    assert "Compréhension du contexte basse" in prompt


def test_follow_up_prompt_allows_general_knowledge_when_source_is_insufficient():
    prompt = build_follow_up_prompt(
        "Tableau de résultats de détection par classe.",
        "Comment une classe peut-elle ne pas être la bonne ?",
    )

    assert "complète avec tes connaissances générales" in prompt
    assert "N'écris pas simplement \"impossible de répondre\"" in prompt
    assert "distingue la source locale et l'explication externe" in prompt


def test_parse_rephrasing():
    raw = {
        "rephrasing_angle": "analogie",
        "rephrased_paragraph": "On peut voir ce résultat comme un changement de point de vue.",
        "note": "Compare les deux formulations.",
    }

    assert parse_rephrasing(raw)["rephrasing_angle"] == "analogie"


def test_parse_flashcard_invalid_difficulty():
    assert parse_flashcard({"front": "A", "back": "B", "tags": [], "difficulty": 4}) is None


def test_parse_session_summary():
    raw = {
        "session_summary": {
            "duration_s": 60,
            "paragraphs_read": 3,
            "flashcards_created": 1,
            "rephrasings_count": 0,
            "success_rate": 0.8,
            "qualitative_summary": "Session régulière.",
            "metacognitive_questions": [
                "Qu'est-ce qui a aidé ?",
                "Où était le blocage ?",
                "Que reste-t-il à clarifier ?",
            ],
        }
    }

    assert parse_session_summary(raw)["session_summary"]["success_rate"] == 0.8


def test_parse_meta_cognition_questions_exactly_three():
    parsed = parse_meta_cognition_questions({
        "questions": [
            "Où as-tu progressé ?",
            "Qu'est-ce qui t'a bloqué ?",
            "Quelle stratégie as-tu utilisée ?",
        ]
    })

    assert len(parsed["questions"]) == 3


def test_parse_meta_cognition_analysis():
    parsed = parse_meta_cognition_analysis({
        "score_delta": 6,
        "score": 62,
        "reasoning": "Réponses précises.",
        "detected_signals": {
            "awareness_of_difficulties": 0.8,
            "strategy_identification": 0.7,
            "self_evaluation": 0.6,
            "specificity": 0.9,
            "honesty_or_depth": 0.8,
        },
    })

    assert parsed["score_delta"] == 6.0
    assert parsed["detected_signals"]["specificity"] == 0.9


def test_parse_flashcard_tags_normalizes_tags():
    parsed = parse_flashcard_tags({"tags": [" Analyse ", "Important", "Suites", "analyse"]})

    assert parsed["tags"] == ["analyse", "suites"]


def test_parse_chapter_summary():
    raw = {
        "chapter_summary": {
            "title": "Chapitre 1",
            "overview": "Le chapitre introduit l'idée principale.",
            "recap_qa": [
                {"question": "Q1 ?", "answer": "A1."},
                {"question": "Q2 ?", "answer": "A2."},
                {"question": "Q3 ?", "answer": "A3."},
            ],
        }
    }

    assert parse_chapter_summary(raw)["chapter_summary"]["title"] == "Chapitre 1"


def test_parse_curiosity_hook():
    raw = {
        "curiosity_hook": "Ce chapitre va relier une idée simple à une conséquence inattendue.",
        "tone": "intriguing",
        "link_with_chapter": "Il introduit la notion centrale.",
        "estimated_accessibility": 0.7,
    }

    parsed = parse_curiosity_hook(raw)

    assert parsed["tone"] == "intriguing"
    assert parsed["estimated_accessibility"] == 0.7
