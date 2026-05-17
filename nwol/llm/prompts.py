# llm/prompts.py — Prompts JSON pour MetaC-App (bilingue FR/EN)
from __future__ import annotations

import json
import random

import i18n as _i18n


def _t(fr: str, en: str) -> str:
    """Return fr or en string based on current UI language."""
    return en if _i18n.current_lang() == "en" else fr


QUESTION_TYPE_GUIDE: tuple[tuple[str, str, str, str], ...] = (
    (
        "qcm",
        "QCM",
        "vérification rapide de compréhension factuelle",
        "Choisis la bonne proposition parmi 3 ou 4 réponses.",
    ),
    (
        "open",
        "Question ouverte",
        "expression libre, reformulation personnelle",
        "Résume en une phrase l'idée principale du passage.",
    ),
    (
        "comprehension",
        "Question de compréhension textuelle",
        "extraction d'une information explicitement donnée",
        "Quelle définition est donnée pour cette notion ?",
    ),
    (
        "application",
        "Question d'application",
        "mise en pratique sur un calcul, un exemple numérique ou un cas particulier",
        "Applique la relation du passage à ce petit cas.",
    ),
    (
        "curiosity",
        "Question de curiosité / inductive",
        "création d'un déséquilibre cognitif qui pousse à chercher le pourquoi",
        "T'es-tu déjà demandé comment cette idée peut rester vraie dans ce cas ?",
    ),
    (
        "visualization",
        "Exercice de visualisation",
        "vision dans l'espace, schéma mental, représentation d'un mécanisme",
        "Trace mentalement la situation : que vois-tu changer ?",
    ),
    (
        "metacognition",
        "Question métacognitive",
        "prise de conscience du raisonnement utilisé",
        "Comment as-tu trouvé ta réponse ? Qu'as-tu modifié dans ton raisonnement ?",
    ),
    (
        "anticipation",
        "Anticipation / auto-évaluation",
        "surveillance de la compréhension et repérage des difficultés possibles",
        "Qu'est-ce qui pourrait te poser problème ici ?",
    ),
)

QUESTION_TYPE_GUIDE_EN: tuple[tuple[str, str, str, str], ...] = (
    (
        "qcm",
        "MCQ",
        "quick factual comprehension check",
        "Choose the correct answer from 3 or 4 options.",
    ),
    (
        "open",
        "Open question",
        "free expression, personal reformulation",
        "Summarize the main idea of the passage in one sentence.",
    ),
    (
        "comprehension",
        "Reading comprehension",
        "extraction of explicitly stated information",
        "What definition is given for this concept?",
    ),
    (
        "application",
        "Application question",
        "practice on a calculation, numerical example, or specific case",
        "Apply the formula from the passage to this small case.",
    ),
    (
        "curiosity",
        "Curiosity / inductive question",
        "creating cognitive imbalance to push the learner to seek the why",
        "Have you ever wondered how this idea can hold true in this case?",
    ),
    (
        "visualization",
        "Visualization exercise",
        "spatial vision, mental diagram, representation of a mechanism",
        "Mentally trace the situation: what do you see changing?",
    ),
    (
        "metacognition",
        "Metacognitive question",
        "awareness of the reasoning process used",
        "How did you arrive at your answer? What did you adjust in your reasoning?",
    ),
    (
        "anticipation",
        "Anticipation / self-assessment",
        "monitoring comprehension and spotting possible difficulties ahead",
        "What might be challenging for you here?",
    ),
)


def build_question_prompt(
    paragraph: str,
    chapter_title: str = "",
    doc_title: str = "",
    metacog_profile: dict | None = None,
    history: list[dict] | None = None,
    session_gauges: dict | None = None,
    recent_question_types: list[str] | None = None,
    preferred_question_type: str | None = None,
    source_block_id: str | None = None,
    has_existing_question: bool = False,
    standalone: bool = False,
) -> str:
    history = history or []
    session_gauges = session_gauges or {}
    recent_question_types = _normalize_recent_question_types(
        recent_question_types or _question_types_from_history(history)
    )
    adaptation = _question_adaptation(
        paragraph=paragraph,
        gauges=session_gauges,
        recent_question_types=recent_question_types,
        preferred_question_type=preferred_question_type,
        has_existing_question=has_existing_question,
        standalone=standalone,
    )
    sid = source_block_id or ""

    if history:
        history_instruction = _t(
            "\nQuand c'est pédagogiquement pertinent, fais un lien explicite avec une réponse précédente"
            " de l'étudiant, par exemple : \"Tu avais dit quelques paragraphes plus tôt que...\"."
            " Le lien doit aider l'étudiant à consolider sa compréhension, sans le culpabiliser.",
            "\nWhen pedagogically relevant, make an explicit link to a previous student answer,"
            " for example: \"Earlier you mentioned that...\"."
            " The link should help the student consolidate their understanding, without making them feel guilty.",
        )
    else:
        history_instruction = ""

    if standalone:
        question_instruction = _t(
            "Tu reçois une question issue d'une session de lecture, avec sa réponse attendue. "
            "Reformule-la en une question de révision totalement autonome, compréhensible sans "
            "avoir lu le document source. "
            "Remplace impérativement toute formule contextuelle ('selon le passage', "
            "'d'après ce texte', 'dans ce paragraphe', 'le passage', 'd'après le texte') "
            "par le concept ou la donnée précise. "
            "Exemple : 'Selon le passage, qu'est-ce qu'une suite $u$ ?' → "
            "'Donne la définition d'une suite numérique $u_n$.' "
            "Choisis le question_type le plus adapté au contenu (qcm si possible).",
            "You receive a question from a reading session with its expected answer. "
            "Rephrase it as a fully standalone review question, understandable without having read the source document. "
            "You must replace any contextual phrasing ('according to the passage', 'based on this text', "
            "'in this paragraph', 'the passage', 'from the text') with the precise concept or data. "
            "Example: 'According to the passage, what is a sequence $u$?' → "
            "'Give the definition of a numerical sequence $u_n$.' "
            "Choose the most suitable question_type for the content (qcm if possible).",
        )
    elif has_existing_question:
        question_instruction = _t(
            "Le paragraphe contient déjà une ou plusieurs questions. "
            "Demande à l'étudiant d'y répondre directement. "
            "Tu peux ajouter UNE question complémentaire si c'est pédagogiquement pertinent, "
            "mais la question principale doit être celle du texte d'origine. "
            "Choisis question_type selon la forme de la question déjà présente ; "
            "si elle ne correspond à aucun type précis, utilise \"comprehension\".",
            "The paragraph already contains one or more questions. "
            "Ask the student to answer them directly. "
            "You may add ONE supplementary question if pedagogically relevant, "
            "but the main question must be the one from the original text. "
            "Choose question_type based on the form of the existing question; "
            "if it does not match any specific type, use \"comprehension\".",
        )
    else:
        question_instruction = _t(
            "Choisis d'abord UN type pédagogique dans la liste ci-dessous, puis génère UNE question "
            "obligatoire adaptée à ce type. Ne choisis pas toujours \"comprehension\" : varie selon "
            "le paragraphe, le profil et l'effort d'apprentissage le plus utile.",
            "First choose ONE pedagogical type from the list below, then generate ONE required question "
            "adapted to that type. Do not always choose \"comprehension\": vary based on the paragraph, "
            "the profile, and the most useful learning effort.",
        )

    _na = _t("non renseigné", "not specified")
    _types_str = ", ".join(f'"{item[0]}"' for item in QUESTION_TYPE_GUIDE)

    if _i18n.current_lang() == "en":
        _standalone_constraint = (
            "- The question must be understandable without any source document: never write "
            "'according to the passage', 'based on this text', or any contextual reference."
            if standalone else
            "- The question must depend on the provided paragraph, not on external knowledge."
        )
        return f"""You are the adaptive learning companion of MetaC-App.

Context:
- Document: {doc_title or _na}
- Chapter: {chapter_title or _na}
- Metacognitive profile: {_json(metacog_profile or {})}
- Current session gauges: {_json(session_gauges)}
- Recent question types: {_json(recent_question_types)}
- Last 5 session answers: {_json(history)}

Paragraph to assess:
---
{paragraph[:3500]}
---

Available question types:
{_question_type_guide()}

Type selection rules:
- Select exactly one question_type value from: {_types_str}.
- For a definition or dense fact, prefer "qcm" or "comprehension".
- For a central idea to reformulate, prefer "open".
- For a formula, calculation, example, table, or specific case, prefer "application".
- For a figure, diagram, spatial relation, or process to visualize, prefer "visualization".
- To provoke an intuition or hypothesis from the paragraph, prefer "curiosity".
- To make the student articulate their reasoning strategy, prefer "metacognition".
- To anticipate a difficulty, uncertainty, or risk of error, prefer "anticipation".
- The chosen type must stay faithful to the paragraph: do not require external knowledge to answer.
- For "curiosity", the question may open a lead, but the expected answer must remain grounded in the passage.
- For "metacognition" and "anticipation", expected_answer describes elements expected in a good answer, not a single solution.

Mandatory adaptive rules:
{_adaptive_instruction(adaptation)}

{question_instruction}
Adapt difficulty to the profile without making the question punitive.
{history_instruction}
Respond only in valid JSON, without Markdown, in the exact format:
{{
  "question_type": "qcm" or "open" or "comprehension" or "application" or "curiosity" or "visualization" or "metacognition" or "anticipation",
  "question": "question text",
  "choices": ["A", "B", "C", "D"],
  "expected_answer": "short but precise expected answer",
  "evaluation_criteria": ["validation criterion 1", "validation criterion 2"],
  "session_hint": "",
  "source_block_id": "{sid}",
  "paragraph_mask": {{
    "enabled": false,
    "start_char": 0,
    "end_char": 0,
    "placeholder": "temporarily masked answer"
  }}
}}

Constraints:
- If question_type is not "qcm", choices must be [].
- If question_type is "qcm", choices contains 3 or 4 plausible options and expected_answer indicates the correct one.
- If a target pedagogical type is indicated in the adaptive rules, use that question_type unless clearly incompatible with the paragraph content.
- If session_hint is set, it must be a short sentence helping the student regulate their session, without replacing the question.
- Always write mathematical expressions between $...$ (inline) or $$...$$ (display) in valid LaTeX.
- In JSON, escape each LaTeX backslash with a double backslash: write "$u_n \\\\sim n$", never "$u_n \\sim n$".
- Never remove the backslash from LaTeX commands: write \\text{{u}}_n, not ext{{u}}_n.
- The source text may contain raw Unicode symbols (≠, →, ∞): treat them as mathematical content.
- If the paragraph contains [Table: ...] or a [Table N×M rows×columns] annotation, ask a question about the data or trends in the table.
- If the paragraph mentions [Figure: ...] or [Figure on this page: ...], use the caption to contextualize your question.
- If an image is attached to the request, it corresponds to a PDF crop of the paragraph, a formula, or an adjacent figure: use it to resolve OCR ambiguities and understand the notation.
- paragraph_mask.enabled is true only if masking a short portion of the paragraph genuinely helps the student reason without copying.
- If paragraph_mask.enabled is true, start_char and end_char are exact indices in the provided paragraph.
{_standalone_constraint}"""

    _standalone_constraint_fr = (
        "- La question doit être compréhensible sans aucun document source : n'écris jamais 'selon le passage', "
        "'d'après ce texte' ou toute référence contextuelle."
        if standalone else
        "- La question doit dépendre du paragraphe fourni, pas d'un savoir externe."
    )
    return f"""Tu es le compagnon d'apprentissage adaptatif de MetaC-App.

Contexte :
- Document : {doc_title or _na}
- Chapitre : {chapter_title or _na}
- Profil métacognitif : {_json(metacog_profile or {})}
- Jauges courantes de la session : {_json(session_gauges)}
- Types de questions récents : {_json(recent_question_types)}
- 5 dernières réponses de la session : {_json(history)}

Paragraphe à vérifier :
---
{paragraph[:3500]}
---

Types de questions disponibles :
{_question_type_guide()}

Règles de choix du type :
- Sélectionne exactement une valeur question_type parmi : {_types_str}.
- Pour une définition ou un fait dense, privilégie "qcm" ou "comprehension".
- Pour une idée centrale à reformuler, privilégie "open".
- Pour une formule, un calcul, un exemple, un tableau ou un cas particulier, privilégie "application".
- Pour une figure, un schéma, une relation spatiale ou un processus à se représenter, privilégie "visualization".
- Pour provoquer une intuition ou une hypothèse à partir du paragraphe, privilégie "curiosity".
- Pour faire expliciter la stratégie de réponse, privilégie "metacognition".
- Pour faire repérer à l'avance une difficulté, une incertitude ou un risque d'erreur, privilégie "anticipation".
- Le type choisi doit rester fidèle au paragraphe : n'exige pas de connaissances externes pour répondre.
- Pour "curiosity", la question peut ouvrir une piste, mais la réponse attendue doit rester ancrée dans le passage.
- Pour "metacognition" et "anticipation", expected_answer décrit les éléments attendus dans une bonne réponse, pas une solution unique.

Règles adaptatives obligatoires :
{_adaptive_instruction(adaptation)}

{question_instruction}
Adapte la difficulté au profil sans rendre la question punitive.
{history_instruction}
Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "question_type": "qcm" ou "open" ou "comprehension" ou "application" ou "curiosity" ou "visualization" ou "metacognition" ou "anticipation",
  "question": "texte de la question",
  "choices": ["A", "B", "C", "D"],
  "expected_answer": "réponse attendue courte mais précise",
  "evaluation_criteria": ["critère de validation 1", "critère de validation 2"],
  "session_hint": "",
  "source_block_id": "{sid}",
  "paragraph_mask": {{
    "enabled": false,
    "start_char": 0,
    "end_char": 0,
    "placeholder": "réponse masquée temporairement"
  }}
}}

Contraintes :
- Si question_type ne vaut pas "qcm", choices doit être [].
- Si question_type vaut "qcm", choices contient 3 ou 4 choix plausibles et expected_answer indique le bon choix.
- Si un Type pédagogique cible est indiqué dans les règles adaptatives, utilise ce question_type sauf contradiction manifeste avec le contenu du paragraphe.
- Si session_hint est renseigné, il doit être une phrase courte qui aide l'étudiant à réguler sa session, sans remplacer la question.
- Écris TOUJOURS les expressions mathématiques entre $...$ (inline) ou $$...$$ (display) en LaTeX valide.
- Dans le JSON, échappe chaque backslash LaTeX avec un double backslash : écris "$u_n \\\\sim n$", jamais "$u_n \\sim n$".
- Ne supprime jamais le backslash des commandes LaTeX : écris \\text{{u}}_n, pas ext{{u}}_n.
- Le texte source peut contenir des symboles Unicode bruts (≠, →, ∞) : traite-les comme du contenu mathématique.
- Si le paragraphe contient [Tableau: ...] ou une annotation [Tableau N×M lignes×colonnes], pose une question sur les données ou les tendances du tableau.
- Si le paragraphe mentionne [Figure: ...] ou [Figure sur cette page : ...], utilise la légende pour contextualiser ta question.
- Si une image est jointe à la requête, elle correspond à un crop PDF du paragraphe, d'une formule, ou d'une figure adjacente : utilise-la pour lever les ambiguïtés OCR et comprendre les notations.
- paragraph_mask.enabled vaut true seulement si masquer une courte portion du paragraphe aide vraiment l'étudiant à raisonner sans recopier.
- Si paragraph_mask.enabled vaut true, start_char et end_char sont des indices exacts dans le paragraphe fourni.
{_standalone_constraint_fr}"""


def _question_type_guide() -> str:
    guide = QUESTION_TYPE_GUIDE_EN if _i18n.current_lang() == "en" else QUESTION_TYPE_GUIDE
    sep = "Example:" if _i18n.current_lang() == "en" else "Exemple :"
    return "\n".join(
        f'- "{key}" — {label} : {purpose}. {sep} {example}'
        for key, label, purpose, example in guide
    )


def _question_adaptation(
    paragraph: str,
    gauges: dict,
    recent_question_types: list[str],
    preferred_question_type: str | None,
    has_existing_question: bool,
    standalone: bool,
) -> dict:
    valid_types = tuple(item[0] for item in QUESTION_TYPE_GUIDE)
    explicit = _normalize_question_type(preferred_question_type, valid_types)
    if explicit:
        return {
            "preferred_type": explicit,
            "strategy": _t("type fourni par le contexte appelant", "type provided by calling context"),
            "attention_break": _gauge(gauges, "attention") < 45.0,
            "simplify": _gauge(gauges, "context_comprehension") < 45.0,
        }

    attention = _gauge(gauges, "attention")
    comprehension = _gauge(gauges, "context_comprehension")
    curiosity = _gauge(gauges, "curiosity")
    meta_cognition = _gauge(gauges, "meta_cognition")

    if has_existing_question and not standalone:
        return {
            "preferred_type": "",
            "strategy": _t(
                "répondre à la question déjà présente dans le paragraphe",
                "answer the question already present in the paragraph",
            ),
            "attention_break": attention < 45.0,
            "simplify": comprehension < 45.0,
        }

    if attention < 45.0:
        weights: dict[str, float] = {
            "qcm": 1.0,
            "open": 1.0,
            "comprehension": 1.0,
            "application": 0.85,
            "curiosity": 0.8,
            "visualization": 0.7,
            "metacognition": 1.8,
            "anticipation": 1.2,
        }
        _apply_recent_penalty(weights, recent_question_types)
        preferred = _weighted_question_type(weights)
        return {
            "preferred_type": preferred,
            "strategy": _t("pause_attention", "attention_break"),
            "attention_break": True,
            "simplify": True,
        }

    weights = {
        "qcm": 1.0,
        "open": 1.0,
        "comprehension": 1.0,
        "application": 0.85,
        "curiosity": 0.8,
        "visualization": 0.7,
        "metacognition": 0.7,
        "anticipation": 0.7,
    }
    lower = (paragraph or "").lower()
    if "$" in paragraph or "\\" in paragraph or any(sign in paragraph for sign in ("=", "≤", "≥", "≈", "≠", "∑", "∫")):
        weights["application"] += 1.8
    if "[tableau" in lower or "|" in paragraph:
        weights["application"] += 1.5
    if "[figure" in lower or "schéma" in lower or "schema" in lower:
        weights["visualization"] += 2.0

    strategy = _t("diversifier les types de questions", "diversify question types")
    if comprehension < 45.0:
        weights["qcm"] += 2.6
        weights["comprehension"] += 2.2
        weights["application"] *= 0.65
        weights["visualization"] *= 0.75
        strategy = _t(
            "simplifier car la compréhension du contexte est basse",
            "simplify because context comprehension is low",
        )
    if curiosity < 45.0:
        weights["curiosity"] += 3.4
        weights["open"] += 0.4
        strategy = _t("relancer curiosité et créativité", "boost curiosity and creativity")
    if meta_cognition < 38.0:
        weights["metacognition"] += 1.2
        weights["anticipation"] += 0.5
        strategy = _t("renforcer la méta-cognition", "strengthen metacognition")

    # cap de fréquence : pénalise fortement la métacognition si posée récemment
    if any(t in ("metacognition", "anticipation") for t in recent_question_types[-2:]):
        weights["metacognition"] *= 0.12
        weights["anticipation"] *= 0.25

    _apply_recent_penalty(weights, recent_question_types)
    preferred = _weighted_question_type(weights)
    return {
        "preferred_type": preferred,
        "strategy": strategy,
        "attention_break": False,
        "simplify": comprehension < 45.0,
    }


def _adaptive_instruction(adaptation: dict) -> str:
    preferred = adaptation.get("preferred_type") or ""
    default_strategy = _t("diversifier les questions", "diversify question types")
    lines = [
        f"- {_t('Stratégie', 'Strategy')} : {adaptation.get('strategy') or default_strategy}.",
    ]
    if preferred:
        lines.append(
            f'- {_t("Type pédagogique cible", "Target pedagogical type")} : "{preferred}".'
        )
    if adaptation.get("attention_break"):
        lines.append(
            _t(
                "- Attention actuelle sous le seuil 45 : renseigne session_hint avec une suggestion "
                "explicite de pause courte avant de continuer, puis pose une question très légère.",
                "- Current attention below threshold 45: set session_hint with an explicit suggestion "
                "for a short break before continuing, then ask a very light question.",
            )
        )
    if adaptation.get("simplify"):
        lines.append(
            _t(
                "- Compréhension du contexte basse : formule une question simple, en une étape, "
                "avec une réponse attendue courte et concrète.",
                "- Context comprehension is low: formulate a simple, single-step question "
                "with a short and concrete expected answer.",
            )
        )
    lines.append(
        _t(
            "- Assure une vraie diversité sur la session : évite de répéter le même question_type "
            "quand le contenu permet un autre type pertinent.",
            "- Ensure genuine diversity across the session: avoid repeating the same question_type "
            "when the content allows another relevant type.",
        )
    )
    return "\n".join(lines)


def _question_types_from_history(history: list[dict]) -> list[str]:
    result: list[str] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        qtype = item.get("question_type")
        if isinstance(qtype, str) and qtype:
            result.append(qtype)
    return result


def _normalize_recent_question_types(values: list[str]) -> list[str]:
    valid = {item[0] for item in QUESTION_TYPE_GUIDE}
    return [value for value in (_normalize_question_type(v, tuple(valid)) for v in values or []) if value]


def _normalize_question_type(value: str | None, valid_types: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip().lower().replace("é", "e").replace("è", "e")
    aliases = {
        "visualisation": "visualization",
        "metacognition": "metacognition",
        "meta_cognition": "metacognition",
        "curiosite": "curiosity",
        "question_ouverte": "open",
        "comprehension_textuelle": "comprehension",
    }
    token = aliases.get(token, token)
    return token if token in valid_types else ""


def _apply_recent_penalty(weights: dict[str, float], recent_question_types: list[str]) -> None:
    for index, qtype in enumerate(reversed(recent_question_types[-4:]), start=1):
        if qtype in weights:
            weights[qtype] *= 0.18 if index == 1 else 0.45


def _weighted_question_type(weights: dict[str, float]) -> str:
    items = [(key, max(0.05, float(value))) for key, value in weights.items()]
    total = sum(value for _key, value in items)
    threshold = random.random() * total
    cumulative = 0.0
    for key, weight in items:
        cumulative += weight
        if threshold <= cumulative:
            return key
    return items[-1][0]


def _gauge(gauges: dict, key: str) -> float:
    try:
        return float((gauges or {}).get(key, 50.0))
    except (TypeError, ValueError):
        return 50.0


def build_evaluation_prompt(
    question: dict,
    user_answer: str,
    paragraph: str,
    metacog_profile: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    history = history or []
    if history:
        history_instruction = _t(
            "\nSi une réponse antérieure aide à expliquer le verdict ou le feedback, référence-la explicitement"
            " avec tact (\"Tu avais déjà repéré...\", \"Tu avais affirmé plus tôt...\") et montre le lien logique.",
            "\nIf a previous answer helps explain the verdict or feedback, reference it explicitly"
            " and tactfully (\"You had already noticed...\", \"You mentioned earlier...\") and show the logical link.",
        )
    else:
        history_instruction = ""

    question_type = (question or {}).get("question_type", "")
    if question_type in ("metacognition", "anticipation"):
        _flashcard_constraint = _t(
            "- flashcard DOIT être null : les questions de type métacognitif et d'anticipation "
            "ne génèrent jamais de flashcard.",
            "- flashcard MUST be null: metacognitive and anticipation question types never generate a flashcard.",
        )
        _flashcard_example = "null"
    else:
        _flashcard_constraint = _t(
            "- flashcard : fournis TOUJOURS une flashcard. "
            "front doit être une question autonome, compréhensible sans avoir lu le document : "
            "si la question fait référence au passage ('selon le passage', 'd'après ce texte'…), "
            "remplace cette référence par le concept ou la donnée précise tirée du paragraphe — "
            "intègre le contexte dans la logique même de la question, pas en préambule. "
            "Exemple : 'Selon le passage, qu\\'est-ce qu\\'une suite ?' → 'Donne la définition d\\'une suite numérique $u_n$.' "
            "back doit être la réponse attendue concise, fidèle à expected_answer.",
            "- flashcard: ALWAYS provide a flashcard. "
            "front must be a standalone question, understandable without having read the document: "
            "if the question references the passage ('according to the passage', 'based on this text'…), "
            "replace that reference with the precise concept or data from the paragraph — "
            "embed the context into the logic of the question itself, not as a preamble. "
            "Example: 'According to the passage, what is a sequence?' → 'Give the definition of a numerical sequence $u_n$.' "
            "back must be the concise expected answer, faithful to expected_answer.",
        )
        _flashcard_example = (
            '{"front": "standalone reformulated question", "back": "concise answer",'
            ' "tags": ["tag1", "tag2"], "difficulty": 2}'
        )

    if _i18n.current_lang() == "en":
        return f"""You are evaluating a student's answer in MetaC-App.

Metacognitive profile: {_json(metacog_profile or {})}
Recent session history: {_json(history)}

Source paragraph:
---
{paragraph[:3500]}
---

Question:
{_json(question)}

Student's answer:
---
{user_answer[:2000]}
---

Evaluate strictly against the paragraph and the expected answer.
{history_instruction}
Respond only in valid JSON, without Markdown. Keep the output concise:
{{
  "verdict": "partial",
  "feedback": "brief and useful feedback",
  "completion": "element to add if partial, otherwise empty string",
  "hint": "hint if incorrect, otherwise empty string",
  "metacog_signals": {{
    "context_comprehension": 0.0,
    "creativity": 0.0,
    "attention": 0.0,
    "retention": 0.0,
    "curiosity": 1.0,
    "meta_cognition": 0.0
  }},
  "curiosity_signals": {{
    "asked_follow_up_question": false,
    "asked_for_clarification": false,
    "asked_for_example": false,
    "explored_beyond_required_answer": false
  }},
  "creativity_signals": {{
    "goes_beyond_prompt": false,
    "makes_connections": false,
    "uses_analogy": false,
    "personal_reformulation": false,
    "original_hypothesis": false,
    "depth_of_reflection": 0.0
  }},
  "answer_to_user_question": null,
  "flashcard": {_flashcard_example}
}}

Constraints:
- Metacognitive signals are between -2.0 and 2.0.
- verdict must be exactly "correct", "partial", or "incorrect".
- meta_cognition stays at 0.0 here: it will only be evaluated in the end-of-session debrief.
- verdict is "correct" if the expected idea is present, even if the student adds a personal reflection, a caveat, or a question that does not contradict the paragraph.
- verdict is "partial" only if the main idea is present but too imprecise or incomplete.
- verdict is "incorrect" only if the main idea is absent, contradicted, or off-topic.
- hint must only be set if verdict is "incorrect". Otherwise hint must be an empty string.
- completion must only be set if verdict is "partial". Otherwise completion must be an empty string.
- If verdict is "correct", feedback must be a short, specific sentence validating what the student understood — cite a precise element from their answer or the paragraph. Never write simply "Correct" or "Good answer": always add a nuance, a link, or a useful pedagogical remark.
- If the student's answer contains a follow-up question, a clarification request, an example request, or a request for deeper understanding, increase the curiosity signal and set curiosity_signals.
- Do not answer this question in answer_to_user_question during evaluation: set answer_to_user_question to null. The follow-up question will be handled by the dedicated "Ask a question about this paragraph" field.
- If the answer goes beyond the minimum expected, makes connections, reformulates in their own words, proposes an analogy or a hypothesis, increase the creativity signal and set creativity_signals.
- If an image is attached to the request, it corresponds to a PDF crop of the paragraph or an adjacent formula: use it only to confirm the notation and stay faithful to the source.
{_flashcard_constraint}
- Do not give the complete solution directly if verdict is incorrect."""

    return f"""Tu évalues une réponse d'étudiant dans MetaC-App.

Profil métacognitif : {_json(metacog_profile or {})}
Historique récent de la session : {_json(history)}

Paragraphe source :
---
{paragraph[:3500]}
---

Question :
{_json(question)}

Réponse de l'étudiant :
---
{user_answer[:2000]}
---

Évalue strictement par rapport au paragraphe et à la réponse attendue.
{history_instruction}
Réponds uniquement en JSON valide, sans Markdown. Garde la sortie courte :
{{
  "verdict": "partial",
  "feedback": "retour bref et utile",
  "completion": "élément à ajouter si partial, sinon chaîne vide",
  "hint": "indice si incorrect, sinon chaîne vide",
  "metacog_signals": {{
    "context_comprehension": 0.0,
    "creativity": 0.0,
    "attention": 0.0,
    "retention": 0.0,
    "curiosity": 1.0,
    "meta_cognition": 0.0
  }},
  "curiosity_signals": {{
    "asked_follow_up_question": false,
    "asked_for_clarification": false,
    "asked_for_example": false,
    "explored_beyond_required_answer": false
  }},
  "creativity_signals": {{
    "goes_beyond_prompt": false,
    "makes_connections": false,
    "uses_analogy": false,
    "personal_reformulation": false,
    "original_hypothesis": false,
    "depth_of_reflection": 0.0
  }},
  "answer_to_user_question": null,
  "flashcard": {_flashcard_example}
}}

Contraintes :
- Les signaux métacognitifs sont entre -2.0 et 2.0.
- verdict doit être exactement "correct", "partial" ou "incorrect".
- meta_cognition reste à 0.0 ici : elle sera évaluée uniquement dans le sas de fin de session.
- verdict vaut "correct" si l'idée attendue est présente, même si l'étudiant ajoute une réflexion personnelle, une réserve ou une question qui ne contredit pas le paragraphe.
- verdict vaut "partial" seulement si l'idée principale est présente mais trop imprécise ou incomplète.
- verdict vaut "incorrect" seulement si l'idée principale est absente, contredite ou hors sujet.
- hint doit être renseigné uniquement si verdict vaut "incorrect". Sinon hint doit être une chaîne vide.
- completion doit être renseigné uniquement si verdict vaut "partial". Sinon completion doit être une chaîne vide.
- Si verdict vaut "correct", feedback doit être une phrase courte et spécifique qui valide ce que l'étudiant a bien saisi — cite un élément précis de sa réponse ou du paragraphe. Ne jamais écrire simplement "Correct" ou "Bonne réponse" : ajoute toujours une nuance, un lien ou une remarque pédagogique utile.
- Si la réponse de l'étudiant contient une question de suivi, une demande de clarification, d'exemple ou d'approfondissement, augmente le signal curiosity et renseigne curiosity_signals.
- Ne réponds pas à cette question dans answer_to_user_question pendant l'évaluation : mets answer_to_user_question à null. La question de suivi sera traitée par le champ dédié "Poser une question sur ce paragraphe".
- Si la réponse dépasse le minimum attendu, fait des liens, reformule avec ses mots, propose une analogie ou une hypothèse, augmente le signal creativity et renseigne creativity_signals.
- Si une image est jointe à la requête, elle correspond à un crop PDF du paragraphe ou d'une formule adjacente : utilise-la seulement pour confirmer les notations et rester fidèle à la source.
{_flashcard_constraint}
- Ne donne pas directement la solution complète si verdict vaut incorrect."""


def build_follow_up_prompt(
    paragraph: str,
    user_question: str,
    metacog_profile: dict | None = None,
) -> str:
    if _i18n.current_lang() == "en":
        return f"""You are the adaptive learning companion of MetaC-App.

The student has asked a follow-up question about a paragraph they just read.

Metacognitive profile: {_json(metacog_profile or {})}

Source paragraph:
---
{paragraph[:3500]}
---

Student's question:
---
{user_question[:500]}
---

Respond in valid JSON, without Markdown:
{{
  "answer": "clear and pedagogical answer: first what the paragraph says, then a general explanation if useful",
  "metacog_signals": {{
    "context_comprehension": 0.0,
    "creativity": 0.0,
    "attention": 0.0,
    "retention": 0.0,
    "curiosity": 0.0,
    "meta_cognition": 0.0
  }},
  "curiosity_signals": {{
    "asked_follow_up_question": true,
    "asked_for_clarification": false,
    "asked_for_example": false,
    "explored_beyond_required_answer": false
  }}
}}

Constraints:
- curiosity must be at least 1.0 because the student is asking a follow-up question.
- Always answer the student's question when a useful answer is possible.
- If the paragraph does not provide enough information, supplement with your general knowledge.
- Structure your answer in two parts: (1) what the paragraph says on the subject, (2) what your general knowledge adds — even if the paragraph already addresses the question.
- The general knowledge supplement is MANDATORY whenever the student's question goes beyond the strict definition in the paragraph (examples, special cases, properties, counterexamples, applications…). Introduce it with "More generally, ..." or "In mathematics / in physics / in computer science, ...".
- Do not invent precise facts about the document if the paragraph does not provide them; distinguish the local source from the external explanation.
- Do not simply write "unable to answer": if the paragraph does not answer, use your knowledge to explain the concept.
- meta_cognition stays at 0.0."""

    return f"""Tu es le compagnon d'apprentissage adaptatif de MetaC-App.

L'étudiant a posé une question de suivi sur un paragraphe qu'il vient de lire.

Profil métacognitif : {_json(metacog_profile or {})}

Paragraphe source :
---
{paragraph[:3500]}
---

Question de l'étudiant :
---
{user_question[:500]}
---

Réponds en JSON valide, sans Markdown :
{{
  "answer": "réponse claire et pédagogique : d'abord ce que dit le paragraphe, puis une explication générale si utile",
  "metacog_signals": {{
    "context_comprehension": 0.0,
    "creativity": 0.0,
    "attention": 0.0,
    "retention": 0.0,
    "curiosity": 0.0,
    "meta_cognition": 0.0
  }},
  "curiosity_signals": {{
    "asked_follow_up_question": true,
    "asked_for_clarification": false,
    "asked_for_example": false,
    "explored_beyond_required_answer": false
  }}
}}

Contraintes :
- curiosity doit être au moins 1.0 car l'étudiant pose une question de suivi.
- Réponds toujours à la question de l'étudiant quand une réponse utile est possible.
- Si le paragraphe ne donne pas assez d'éléments, complète avec tes connaissances générales.
- Structure ta réponse en deux temps : (1) ce que le paragraphe dit sur le sujet, (2) ce que tes connaissances générales apportent en complément — même si le paragraphe aborde déjà la question.
- Le complément de connaissances générales est OBLIGATOIRE dès que la question de l'étudiant dépasse la définition stricte du paragraphe (exemples, cas particuliers, propriétés, contre-exemples, applications…). Introduis-le avec "Plus généralement, ..." ou "En mathématiques / en physique / en informatique, ...".
- N'invente pas de fait précis sur le document si le paragraphe ne le donne pas ; distingue la source locale et l'explication externe.
- N'écris pas simplement "impossible de répondre" : si le paragraphe ne répond pas, utilise tes connaissances pour expliquer le concept.
- meta_cognition reste à 0.0."""


def build_rephrasing_prompt(paragraph: str, attempt_count: int) -> str:
    if _i18n.current_lang() == "en":
        return f"""You are rephrasing a course paragraph to unblock a student.

Number of consecutive incorrect attempts: {attempt_count}

Original paragraph:
---
{paragraph[:3500]}
---

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "rephrasing_angle": "chosen angle",
  "rephrased_paragraph": "clear, faithful, and more accessible reformulation",
  "note": "brief sentence on what to look at differently"
}}

Constraints:
- Do not simplify to the point of changing the content.
- Keep important formulas and notation.
- If an image is attached to the request, it corresponds to the PDF crop of the passage: use it to preserve notation that OCR may have degraded."""

    return f"""Tu reformules un paragraphe de cours pour débloquer un étudiant.

Nombre de tentatives incorrectes consécutives : {attempt_count}

Paragraphe original :
---
{paragraph[:3500]}
---

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "rephrasing_angle": "angle choisi",
  "rephrased_paragraph": "reformulation claire, fidèle et plus accessible",
  "note": "phrase brève sur ce qu'il faut regarder autrement"
}}

Contraintes :
- Ne simplifie pas au point de changer le contenu.
- Garde les formules et notations importantes.
- Si une image est jointe à la requête, elle correspond au crop PDF du passage : utilise-la pour préserver les notations que l'OCR aurait pu dégrader."""


def build_session_summary_prompt(
    session_data: dict,
    metacog_profile: dict | None = None,
    session_gauges: dict | None = None,
) -> str:
    profile = metacog_profile or {}
    gauges = session_gauges or {}

    is_en = _i18n.current_lang() == "en"

    gauge_labels = {
        "attention": _t("Attention", "Attention"),
        "context_comprehension": _t("Compréhension", "Comprehension"),
        "creativity": _t("Créativité", "Creativity"),
        "retention": _t("Rétention", "Retention"),
        "curiosity": _t("Curiosité", "Curiosity"),
        "meta_cognition": _t("Métacognition", "Metacognition"),
    }
    gauge_lines = []
    for key, label in gauge_labels.items():
        session_val = gauges.get(key)
        profile_val = profile.get(key)
        if session_val is not None and profile_val is not None:
            diff = float(session_val) - float(profile_val)
            if is_en:
                trend = (
                    f"+{diff:.1f} (exceeded)" if diff >= 8
                    else (f"{diff:.1f} (below baseline)" if diff <= -8 else f"{diff:+.1f} (stable)")
                )
            else:
                trend = (
                    f"+{diff:.1f} (surpassé)" if diff >= 8
                    else (f"{diff:.1f} (en retrait)" if diff <= -8 else f"{diff:+.1f} (stable)")
                )
            gauge_lines.append(
                f"  {label}: session={float(session_val):.1f} | "
                f"{_t('profil', 'profile')}={float(profile_val):.1f} | "
                f"{_t('écart', 'delta')}={trend}"
            )
        elif session_val is not None:
            gauge_lines.append(f"  {label}: session={float(session_val):.1f}")
    gauge_comparison = "\n".join(gauge_lines) if gauge_lines else f"  ({_t('non disponibles', 'not available')})"

    stats_only = {k: v for k, v in session_data.items() if k not in ("profile", "gauges", "session_score")}

    if is_en:
        return f"""You are producing the end-of-session debrief for MetaC-App.

Session statistics:
{_json(stats_only)}

Session gauges vs reference profile comparison (scale 0–100):
{gauge_comparison}

Reference metacognitive profile (historical averages):
{_json(profile)}

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "session_summary": {{
    "duration_s": 0,
    "paragraphs_read": 0,
    "flashcards_created": 0,
    "rephrasings_count": 0,
    "success_rate": 0.0,
    "qualitative_summary": "...",
    "metacognitive_questions": [
      "question 1",
      "question 2",
      "question 3"
    ]
  }}
}}

Constraints:
- success_rate is between 0.0 and 1.0.
- qualitative_summary: 2 to 3 sentences in English. Mention at least one positive point, one area for improvement, and one concrete suggestion. If a gauge exceeds its profile by ≥8 pts, explicitly note this (e.g., "your attention was noticeably above your usual level"). If a gauge is below by ≥8 pts, note that too. Be precise and personalized.
- Provide exactly 3 short, clear, and distinct metacognitive questions, adapted to the session data and gauges."""

    return f"""Tu produis le sas de sortie d'une session MetaC-App.

Statistiques de session :
{_json(stats_only)}

Comparaison jauges session vs profil de référence (échelle 0–100) :
{gauge_comparison}

Profil métacognitif de référence (moyennes historiques) :
{_json(profile)}

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "session_summary": {{
    "duration_s": 0,
    "paragraphs_read": 0,
    "flashcards_created": 0,
    "rephrasings_count": 0,
    "success_rate": 0.0,
    "qualitative_summary": "...",
    "metacognitive_questions": [
      "question 1",
      "question 2",
      "question 3"
    ]
  }}
}}

Contraintes :
- success_rate est entre 0.0 et 1.0.
- qualitative_summary : 2 à 3 phrases en français. Mentionne au moins un point positif, un point d'amélioration, et une suggestion concrète. Si une jauge dépasse de ≥8 pts son profil, signale explicitement ce surpassement (ex : "ton attention était nettement au-dessus de ton niveau habituel"). Si une jauge est en retrait de ≥8 pts, signale-le aussi. Sois précis et personnalisé.
- Fournis exactement 3 questions métacognitives courtes, claires et différentes, adaptées aux données et aux jauges de la session."""


def build_meta_cognition_questions_prompt(
    session_summary: dict | None = None,
    recent_user_answers: list[dict] | list[str] | None = None,
    previous_end_questions: list[str] | None = None,
    user_profile: dict | None = None,
) -> str:
    if _i18n.current_lang() == "en":
        return f"""You are generating metacognitive questions for the end of a learning session.

Session summary:
{_json(session_summary or {})}

Recent user answers:
{_json(recent_user_answers or [])}

Questions already asked recently:
{_json(previous_end_questions or [])}

User profile:
{_json(user_profile or {})}

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "questions": ["question 1", "question 2", "question 3"]
}}

Constraints:
- You must produce exactly 3 questions.
- They must help the user reflect on their comprehension, blocks, strategies, and self-assessment.
- The questions must be short, clear, context-adapted, and different from each other.
- Avoid repeating questions that have already been asked if possible.
- Do not ask any question outside the metacognition framework."""

    return f"""Tu génères des questions de méta-cognition pour la fin d'une session d'apprentissage.

Résumé de session :
{_json(session_summary or {})}

Réponses récentes de l'utilisateur :
{_json(recent_user_answers or [])}

Questions déjà posées récemment :
{_json(previous_end_questions or [])}

Profil utilisateur :
{_json(user_profile or {})}

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "questions": ["question 1", "question 2", "question 3"]
}}

Contraintes :
- Tu dois produire exactement 3 questions.
- Elles doivent aider l'utilisateur à réfléchir à sa compréhension, ses blocages, ses stratégies et son auto-évaluation.
- Les questions doivent être courtes, claires, adaptées au contexte et différentes entre elles.
- Évite de reprendre exactement les questions déjà posées si possible.
- Ne pose aucune question hors du cadre de la méta-cognition."""


def build_meta_cognition_analysis_prompt(
    questions: list[str],
    answers: list[str],
    session_context: dict | None = None,
    user_profile: dict | None = None,
) -> str:
    if _i18n.current_lang() == "en":
        return f"""You are analyzing a user's answers to metacognitive questions.

Questions:
{_json(questions)}

User's answers:
{_json(answers)}

Session context:
{_json(session_context or {})}

User profile:
{_json(user_profile or {})}

Evaluate whether the user accurately identifies their difficulties, strategies, comprehension level, and feelings.
Increase the score if the answers are concrete, honest, reflective, and useful.
Decrease the score if the answers are vague, absent, superficial, or off-topic.

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "score_delta": 0.0,
  "score": 50.0,
  "reasoning": "brief reasoning",
  "detected_signals": {{
    "awareness_of_difficulties": 0.0,
    "strategy_identification": 0.0,
    "self_evaluation": 0.0,
    "specificity": 0.0,
    "honesty_or_depth": 0.0
  }}
}}

Constraints:
- score_delta is generally between -12 and +12.
- score and all signals are bounded between 0.0 and 100.0 for score, 0.0 and 1.0 for signals.
- Return a negative score_delta if the answers are absent or too vague."""

    return f"""Tu analyses les réponses de l'utilisateur à des questions de méta-cognition.

Questions :
{_json(questions)}

Réponses de l'utilisateur :
{_json(answers)}

Contexte de session :
{_json(session_context or {})}

Profil utilisateur :
{_json(user_profile or {})}

Évalue si l'utilisateur identifie ses difficultés, ses stratégies, son niveau de compréhension et son ressenti avec précision.
Augmente le score si les réponses sont concrètes, honnêtes, réflexives et utiles.
Diminue le score si les réponses sont vagues, absentes, superficielles ou hors sujet.

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "score_delta": 0.0,
  "score": 50.0,
  "reasoning": "raisonnement bref",
  "detected_signals": {{
    "awareness_of_difficulties": 0.0,
    "strategy_identification": 0.0,
    "self_evaluation": 0.0,
    "specificity": 0.0,
    "honesty_or_depth": 0.0
  }}
}}

Contraintes :
- score_delta est généralement entre -12 et +12.
- score et tous les signaux sont bornés entre 0.0 et 100.0 pour score, 0.0 et 1.0 pour les signaux.
- Retourne score_delta négatif si les réponses sont absentes ou trop vagues."""


def build_flashcard_tags_prompt(
    front: str,
    back: str,
    session_context: dict | None = None,
    existing_sections: list[str] | None = None,
    existing_tags: list[str] | None = None,
) -> str:
    if _i18n.current_lang() == "en":
        return f"""You are generating tags to classify a flashcard.

Front:
---
{(front or "")[:1200]}
---

Back:
---
{(back or "")[:1200]}
---

Available context:
{_json(session_context or {})}

Existing sections:
{_json(existing_sections or [])}

Existing tags:
{_json(existing_tags or [])}

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "tags": ["tag 1", "tag 2"]
}}

Constraints:
- Generate between 2 and 6 short, relevant, duplicate-free tags in English.
- Normalize tags in lowercase.
- Avoid vague tags like "course", "important", or "misc".
- Reuse existing tags or sections when they match the content.
- Do not invent unnecessary categories if an existing tag fits."""

    return f"""Tu génères des tags pour classer une flashcard.

Recto :
---
{(front or "")[:1200]}
---

Verso :
---
{(back or "")[:1200]}
---

Contexte disponible :
{_json(session_context or {})}

Sections existantes :
{_json(existing_sections or [])}

Tags existants :
{_json(existing_tags or [])}

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "tags": ["tag 1", "tag 2"]
}}

Contraintes :
- Génère entre 2 et 6 tags courts, pertinents, sans doublons, en français.
- Normalise les tags en minuscules.
- Évite les tags vagues comme "cours", "important" ou "divers".
- Réutilise les tags ou sections existants lorsqu'ils correspondent au contenu.
- N'invente pas de catégories inutiles si un tag existant convient."""


def build_chapter_summary_prompt(
    chapter_title: str,
    paragraphs_summary: list[dict] | list[str],
    metacog_profile: dict | None = None,
) -> str:
    _na = _t("non renseigné", "not specified")

    if _i18n.current_lang() == "en":
        return f"""You are producing an end-of-chapter summary for MetaC-App.

Chapter: {chapter_title or _na}
Current metacognitive profile (use only to adapt the pedagogical level, never as chapter content):
{_json(metacog_profile or {})}

Elements read in the chapter:
{_json(paragraphs_summary)}

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "chapter_summary": {{
    "title": "chapter title",
    "overview": "short summary in English",
    "recap_qa": [
      {{
        "question": "recap question 1",
        "answer": "concise answer"
      }},
      {{
        "question": "recap question 2",
        "answer": "concise answer"
      }},
      {{
        "question": "recap question 3",
        "answer": "concise answer"
      }}
    ]
  }}
}}

Constraints:
- Give exactly 3 recap Q&As.
- Stay faithful to the elements read in the chapter.
- Never turn the metacognitive profile into course content.
- Formulate answers to help the student verify their understanding."""

    return f"""Tu produis une synthèse de fin de chapitre pour MetaC-App.

Chapitre : {chapter_title or _na}
Profil métacognitif courant (à utiliser seulement pour adapter le niveau pédagogique, jamais comme contenu du chapitre) :
{_json(metacog_profile or {})}

Éléments lus dans le chapitre :
{_json(paragraphs_summary)}

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "chapter_summary": {{
    "title": "titre du chapitre",
    "overview": "synthèse courte en français",
    "recap_qa": [
      {{
        "question": "question récapitulative 1",
        "answer": "réponse concise"
      }},
      {{
        "question": "question récapitulative 2",
        "answer": "réponse concise"
      }},
      {{
        "question": "question récapitulative 3",
        "answer": "réponse concise"
      }}
    ]
  }}
}}

Contraintes :
- Donne exactement 3 Q&R récapitulatives.
- Reste fidèle aux éléments lus dans le chapitre.
- Ne transforme jamais le profil métacognitif en sujet de cours.
- Formule les réponses pour aider l'étudiant à vérifier sa compréhension."""


def build_curiosity_hook_prompt(
    doc_title: str,
    chapter_title: str,
    subchapter_title: str,
    chapter_excerpt: str,
    profile: dict | None = None,
) -> str:
    _na = _t("non renseigné", "not specified")
    no_excerpt = not (chapter_excerpt or "").strip()

    if _i18n.current_lang() == "en":
        excerpt_section = (
            "No excerpt available: base yourself only on the document and chapter title."
            if no_excerpt
            else chapter_excerpt[:2500]
        )
        return f"""You are generating an opening hook for a reader about to read a chapter.

Context:
- Document: {doc_title or _na}
- Chapter: {chapter_title or _na}
- Sub-chapter: {subchapter_title or _na}
- Metacognitive profile: {_json(profile or {})}

Chapter excerpt:
---
{excerpt_section}
---

Write a single opening hook sentence in English, calm and concrete, that makes the reader want to enter this chapter.
The sentence must speak about the document content, not about the reading tool.
Respond only in valid JSON, without Markdown, in the exact format:
{{
  "curiosity_hook": "opening hook sentence",
  "tone": "calm | intriguing | concrete | playful",
  "link_with_chapter": "explicit link with the chapter",
  "estimated_accessibility": 0.0
}}

Constraints:
- curiosity_hook is a single short sentence.
- tone is exactly "calm", "intriguing", "concrete", or "playful".
- estimated_accessibility is between 0.0 and 1.0.
- Never mention the name of the application or the tool."""

    excerpt_section = (
        "Aucun extrait disponible : base-toi uniquement sur le titre du document et du chapitre."
        if no_excerpt
        else chapter_excerpt[:2500]
    )
    return f"""Tu génères une phrase d'accroche pour un lecteur qui s'apprête à lire un chapitre.

Contexte :
- Document : {doc_title or _na}
- Chapitre : {chapter_title or _na}
- Sous-chapitre : {subchapter_title or _na}
- Profil métacognitif : {_json(profile or {})}

Extrait du chapitre :
---
{excerpt_section}
---

Écris une seule phrase d'accroche en français, calme et concrète, qui donne envie d'entrer dans ce chapitre.
La phrase doit parler du contenu du document, pas de l'outil de lecture.
Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "curiosity_hook": "phrase d'accroche",
  "tone": "calm | intriguing | concrete | playful",
  "link_with_chapter": "lien explicite avec le chapitre",
  "estimated_accessibility": 0.0
}}

Contraintes :
- curiosity_hook tient en une phrase courte.
- tone vaut exactement "calm", "intriguing", "concrete" ou "playful".
- estimated_accessibility est entre 0.0 et 1.0.
- Ne mentionne jamais le nom de l'application ou de l'outil."""


def build_latex_paragraph_render_prompt(paragraph_text: str) -> str:
    return f"""Tu es un expert en mise en forme de contenu mathématique extrait de PDFs.

Le texte ci-dessous provient d'un OCR sur un document scientifique. Il peut contenir :
- des formules LaTeX mal extraites (symboles collés, délimiteurs manquants, commandes tronquées)
- du texte prosodique mêlé aux formules
- des artefacts d'extraction

Texte brut extrait :
---
{paragraph_text[:2800]}
---

Ta tâche : produire une version propre et fidèle de ce paragraphe, lisible par un étudiant.

Règles :
- Préserve le sens exact : ne simplifie, ne résume, n'ajoute rien.
- Encadre chaque expression mathématique inline avec $...$ et chaque formule display avec $$...$$.
- N'écris jamais de commande LaTeX brute hors de ces délimiteurs : pas de \\theta, \\cdot, _, ^ ou accolades mathématiques dans le texte courant.
- Préserve les indices, exposants, lettres grecques et noms de variables : ne transforme pas $D_{{meta}}$ en Dmeta, ni $\\hat{{z}}_{{i,j}}$ en z i,j.
- Utilise du LaTeX valide à l'intérieur des délimiteurs.
- Double les backslashes dans le JSON : \\\\frac, \\\\sum, \\\\alpha, etc.
- Le texte non-mathématique reste en français simple, sans Markdown.
- Si une image est jointe, utilise-la pour corriger les notations ambiguës.

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "rendered": "texte nettoyé avec $formules$ correctement délimitées"
}}"""


def build_latex_contextual_chunk_render_prompt(
    target_text: str,
    previous_context: str = "",
    next_context: str = "",
) -> str:
    context = _format_chunk_context(previous_context, next_context)
    return f"""Tu es un expert en mise en forme de contenu mathématique extrait de PDFs.

Le texte cible ci-dessous est un fragment d'une section plus longue. Un contexte voisin peut être fourni uniquement pour comprendre les notations.
{context}

Texte cible à corriger :
---
{target_text[:2200]}
---

Ta tâche : produire une version propre et fidèle du texte cible uniquement.

Règles :
- Ne réécris pas le contexte voisin dans la réponse.
- Préserve le sens exact : ne simplifie, ne résume, n'ajoute rien.
- Encadre chaque expression mathématique inline avec $...$ et chaque formule display avec $$...$$.
- N'écris jamais de commande LaTeX brute hors de ces délimiteurs.
- Préserve les indices, exposants, lettres grecques et noms de variables.
- Utilise du LaTeX valide à l'intérieur des délimiteurs.
- Double les backslashes dans le JSON : \\\\frac, \\\\sum, \\\\alpha, etc.
- Si une image est jointe, utilise-la pour corriger les notations ambiguës.

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "rendered": "texte cible nettoyé avec $formules$ correctement délimitées"
}}"""


def build_latex_paragraph_render_text_prompt(paragraph_text: str) -> str:
    return f"""Tu es un expert en mise en forme de contenu mathématique extrait de PDFs.

Le texte ci-dessous provient d'un OCR sur un document scientifique. Il peut contenir :
- des formules LaTeX mal extraites (symboles collés, délimiteurs manquants, commandes tronquées)
- du texte prosodique mêlé aux formules
- des artefacts d'extraction

Texte brut extrait :
---
{paragraph_text[:2800]}
---

Ta tâche : produire une version propre et fidèle de ce paragraphe, lisible par un étudiant.

Règles :
- Préserve le sens exact : ne simplifie, ne résume, n'ajoute rien.
- Encadre chaque expression mathématique inline avec $...$ et chaque formule display avec $$...$$.
- N'écris jamais de commande LaTeX brute hors de ces délimiteurs : pas de \\theta, \\cdot, _, ^ ou accolades mathématiques dans le texte courant.
- Préserve les indices, exposants, lettres grecques et noms de variables : ne transforme pas $D_{{meta}}$ en Dmeta, ni $\\hat{{z}}_{{i,j}}$ en z i,j.
- Utilise du LaTeX valide à l'intérieur des délimiteurs.
- Le texte non-mathématique reste en français simple, sans Markdown.
- Si une image est jointe, utilise-la pour corriger les notations ambiguës.

Réponds uniquement avec le paragraphe corrigé en texte brut.
N'utilise pas JSON, pas Markdown, pas bloc de code, pas commentaire avant ou après."""


def build_latex_contextual_chunk_render_text_prompt(
    target_text: str,
    previous_context: str = "",
    next_context: str = "",
) -> str:
    context = _format_chunk_context(previous_context, next_context)
    return f"""Tu es un expert en mise en forme de contenu mathématique extrait de PDFs.

Le texte cible ci-dessous est un fragment d'une section plus longue. Un contexte voisin peut être fourni uniquement pour comprendre les notations.
{context}

Texte cible à corriger :
---
{target_text[:2200]}
---

Ta tâche : produire une version propre et fidèle du texte cible uniquement.

Règles :
- Ne réécris pas le contexte voisin dans la réponse.
- Préserve le sens exact : ne simplifie, ne résume, n'ajoute rien.
- Encadre chaque expression mathématique inline avec $...$ et chaque formule display avec $$...$$.
- N'écris jamais de commande LaTeX brute hors de ces délimiteurs.
- Préserve les indices, exposants, lettres grecques et noms de variables.
- Utilise du LaTeX valide à l'intérieur des délimiteurs.
- Si une image est jointe, utilise-la pour corriger les notations ambiguës.

Réponds uniquement avec le texte cible corrigé en texte brut.
N'utilise pas JSON, pas Markdown, pas bloc de code, pas commentaire avant ou après."""


def _format_chunk_context(previous_context: str, next_context: str) -> str:
    parts: list[str] = []
    if previous_context.strip():
        parts.append(f"Contexte précédent, à ne pas réécrire :\n---\n{previous_context[-700:]}\n---")
    if next_context.strip():
        parts.append(f"Contexte suivant, à ne pas réécrire :\n---\n{next_context[:500]}\n---")
    if not parts:
        return ""
    return "\n" + "\n\n".join(parts)


def build_schema_render_prompt(caption: str = "") -> str:
    context = f"\nLégende disponible : {caption}" if caption else ""
    return f"""Tu es un expert en analyse de schémas et diagrammes scientifiques.

Une image d'un schéma ou graphique extrait d'un document PDF t'est fournie.{context}

Ta tâche : produire une description textuelle claire et concise de ce schéma, lisible par un étudiant dans un lecteur de texte.

Règles :
- Décris ce que le schéma représente (type : graphe, diagramme de complexité, arbre, flux logique…).
- Pour un schéma à boîtes et flèches, liste uniquement les libellés visibles dans leur ordre réel.
- N'invente aucune étape, aucun axe, aucune relation et aucun mot qui n'est pas lisible dans l'image.
- Si un libellé est illisible, ignore-le ou dis simplement qu'une étape intermédiaire est illisible.
- Explique les axes, les flèches ou les relations montrées seulement quand ils sont visibles.
- Encadre toute expression mathématique dans $...$ (inline) ou $$...$$ (display).
- Si le schéma montre un ordre de complexité ou une hiérarchie, représente-le avec une notation compacte (ex. : $O(1) \\prec O(\\log n) \\prec O(n) \\prec O(n^2)$).
- Reste concis : 1 à 3 phrases maximum.
- Texte en français, sans Markdown, sans commentaire avant ou après.

Réponds uniquement avec la description en texte brut."""


def build_slide_analysis_prompt() -> str:
    return """Tu es un assistant pédagogique qui analyse des slides de cours universitaires.

Une image d'une slide de présentation t'est fournie.

Ta tâche : produire une analyse pédagogique concise de cette slide, destinée à aider un étudiant à comprendre et retenir son contenu.

Règles :
- Résume le concept principal ou le message clé de la slide en une phrase.
- Identifie les points importants : définitions, formules, étapes clés, exemples, relations entre concepts.
- Si la slide contient une formule mathématique, encadre-la dans $...$ (inline) ou $$...$$ (display).
- Si la slide montre un graphique, un schéma ou un diagramme, décris-en brièvement la structure et ce qu'il illustre.
- Si la slide est un titre ou une slide de transition, dis-le simplement.
- Sois concis : 2 à 5 phrases maximum.
- Texte en français, sans Markdown structuré, sans commentaire avant ou après.

Réponds uniquement avec l'analyse en texte brut."""


def build_table_render_prompt(caption: str = "") -> str:
    context = f"\nTitre ou légende du tableau : {caption}" if caption else ""
    return f"""Tu es un expert en lecture de tableaux scientifiques extraits de PDFs.

Une image d'un tableau extrait d'un document PDF t'est fournie.{context}

Ta tâche : reproduire fidèlement le contenu du tableau en texte structuré lisible.

Règles :
- Présente le tableau avec des colonnes alignées et séparées par des | (format texte simple).
- Inclus la ligne d'en-tête si elle existe.
- Encadre toute expression mathématique dans $...$ (inline).
- Si le tableau est trop large, résume en listant les colonnes et quelques lignes représentatives.
- Texte en français ou selon la langue source, sans Markdown, sans commentaire.

Réponds uniquement avec le tableau en texte brut."""


def build_subject_detection_prompt(doc_title: str, excerpt: str) -> str:
    subjects = "mathématiques, sciences, histoire, géographie, français, informatique, culture"
    safe_excerpt = (excerpt or "").strip()[:700]
    return f"""Tu es un classificateur de cours scolaires et universitaires.

Titre du document : {doc_title or "inconnu"}

Début du document :
---
{safe_excerpt or "Non disponible."}
---

Détermine la matière principale de ce document.
Réponds UNIQUEMENT avec un objet JSON valide, sans markdown, sans commentaire :
{{"subject": "<matière>"}}

La valeur de "subject" DOIT être exactement l'une de : {subjects}
Si tu hésites ou si aucune matière ne correspond clairement, choisis "culture".
Ne mets rien en dehors du JSON."""


def build_quiz_session_analysis_prompt(
    answers_history: list[dict],
    subject_profiles: list[dict] | None = None,
) -> str:
    if _i18n.current_lang() == "en":
        return f"""You are the adaptive learning companion of MetaC-App.
The student has just completed a quiz session.

Answer history (each entry contains: question, user_answer, verdict, score [0.0=incorrect, 0.5=partial, 1.0=correct], category, source, document, chapter_title):
{_json(answers_history)}

Current mastery levels by subject:
{_json(subject_profiles or [])}

Analyze the performance, identify gaps, and recommend specific courses to review.
To recommend a course: use each answer's score to group questions by document/chapter (fields document and chapter_title). Prioritize recommending courses (document + chapter) where the average score is lowest.

Respond only in valid JSON, without Markdown, in the exact format:
{{
  "analysis": "supportive pedagogical summary in 2-3 sentences, factual and direct",
  "weak_subjects": ["subject 1", "subject 2"],
  "courses_to_review": [
    {{
      "title": "title of the course or chapter to review (= document if available, otherwise subject)",
      "subject": "subject key (mathématiques, sciences, histoire, géographie, français, informatique, culture)",
      "reason": "short pedagogical reason based on scores (one sentence)",
      "document": "value of the document field if source=reading, otherwise empty string",
      "chapter_title": "value of the chapter_title field if source=reading, otherwise empty string"
    }}
  ]
}}

Constraints:
- courses_to_review contains between 0 and 3 elements, sorted by ascending average score (lowest first).
- If all answers have score=1.0, courses_to_review must be [].
- Each reason cites the score or number of errors found for this course/chapter.
- Stay factual: base yourself only on the provided history.
- If source=reading and document is not null, use that document as title and set document and chapter_title.
- If source=static, leave document and chapter_title as empty strings."""

    return f"""Tu es le compagnon d'apprentissage adaptatif de MetaC-App.
L'étudiant vient de terminer une session de quiz.

Historique des réponses (chaque entrée contient : question, user_answer, verdict, score [0.0=incorrect, 0.5=partiel, 1.0=correct], category, source, document, chapter_title) :
{_json(answers_history)}

Niveaux de maîtrise actuels par matière :
{_json(subject_profiles or [])}

Analyse les performances, identifie les lacunes et recommande des cours spécifiques à réviser.
Pour recommander un cours : utilise le score de chaque réponse pour regrouper les questions par document/chapitre (champs document et chapter_title). Recommande en priorité les cours (document + chapitre) où le score moyen est le plus faible.

Réponds uniquement en JSON valide, sans Markdown, au format exact :
{{
  "analysis": "synthèse pédagogique bienveillante en 2-3 phrases, factuelle et directe",
  "weak_subjects": ["sujet 1", "sujet 2"],
  "courses_to_review": [
    {{
      "title": "titre du cours ou chapitre à réviser (= document si disponible, sinon matière)",
      "subject": "clé de matière (mathématiques, sciences, histoire, géographie, français, informatique, culture)",
      "reason": "raison courte et pédagogique basée sur les scores (une phrase)",
      "document": "valeur du champ document si source=reading, sinon chaîne vide",
      "chapter_title": "valeur du champ chapter_title si source=reading, sinon chaîne vide"
    }}
  ]
}}

Contraintes :
- courses_to_review contient entre 0 et 3 éléments, triés par score moyen croissant (le plus faible en premier).
- Si toutes les réponses ont score=1.0, courses_to_review doit être [].
- Chaque reason cite le score ou le nombre d'erreurs constatés pour ce cours/chapitre.
- Reste factuel : base-toi uniquement sur l'historique fourni.
- Si source=reading et document non null, utilise ce document comme title et renseigne document et chapter_title.
- Si source=static, laisse document et chapter_title comme chaînes vides."""


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
