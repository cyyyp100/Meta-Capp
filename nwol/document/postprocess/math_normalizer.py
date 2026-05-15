from __future__ import annotations

import re
import unicodedata

from document.models import DocumentBlock


_UNICODE_TO_LATEX: dict[str, str] = {
    # Flèches
    "←": r"\leftarrow",
    "↑": r"\uparrow",
    "→": r"\rightarrow",
    "↓": r"\downarrow",
    "↔": r"\leftrightarrow",
    "↦": r"\mapsto",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "⟹": r"\Longrightarrow",
    "⟺": r"\Longleftrightarrow",
    # Relations et ensembles
    "≠": r"\neq",
    "≈": r"\approx",
    "≅": r"\cong",
    "∼": r"\sim",
    "≃": r"\simeq",
    "≡": r"\equiv",
    "≤": r"\leq",
    "≥": r"\geq",
    "≦": r"\leq",
    "≧": r"\geq",
    "≪": r"\ll",
    "≫": r"\gg",
    "∝": r"\propto",
    "∥": r"\parallel",
    "⊥": r"\perp",
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "⊂": r"\subset",
    "⊆": r"\subseteq",
    "⊄": r"\nsubset",
    "⊃": r"\supset",
    "⊇": r"\supseteq",
    "⊈": r"\nsubseteq",
    "⊉": r"\nsupseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∖": r"\setminus",
    "∀": r"\forall",
    "∃": r"\exists",
    "∄": r"\nexists",
    "∴": r"\therefore",
    "∵": r"\because",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℙ": r"\mathbb{P}",
    # Opérateurs
    "±": r"\pm",
    "∓": r"\mp",
    "×": r"\times",
    "⋅": r"\cdot",
    "·": r"\cdot",
    "÷": r"\div",
    "∞": r"\infty",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∬": r"\iint",
    "∭": r"\iiint",
    "∮": r"\oint",
    "√": r"\sqrt",
    "∛": r"\sqrt[3]",
    "∜": r"\sqrt[4]",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∆": r"\Delta",
    "∧": r"\wedge",
    "∨": r"\vee",
    "¬": r"\neg",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "°": r"^\circ",
    "′": r"'",
    "″": r"''",
    "−": "-",
    # Lettres grecques minuscules
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "ο": "o",
    "π": r"\pi",
    "ρ": r"\rho",
    "ς": r"\varsigma",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "ϕ": r"\varphi",
    "ϵ": r"\varepsilon",
    "ϑ": r"\vartheta",
    # Lettres grecques majuscules
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "ℓ": r"\ell",
    # Suppléments CPGE
    "‖": r"\|",       # double barre (norme)
    "∣": r"|",         # barre divisibilité
    "⌀": r"\emptyset",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "⊕": r"\oplus",
    "⊗": r"\otimes",
    "⊙": r"\odot",
    "⌃": r"\wedge",
    "∐": r"\coprod",
    "⊔": r"\sqcup",
    "⊓": r"\sqcap",
    "≺": r"\prec",
    "≻": r"\succ",
    "⋯": r"\cdots",
    "…": r"\ldots",
    "·": r"\cdot",
    "×": r"\times",
    "÷": r"\div",
    "≤": r"\leq",
    "≥": r"\geq",
    # Indices Unicode
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
    "₊": "_+",
    "₋": "_-",
    "₌": "_=",
    "₍": "_(",
    "₎": "_)",
    "ₐ": "_a",
    "ₑ": "_e",
    "ₕ": "_h",
    "ᵢ": "_i",
    "ⱼ": "_j",
    "ₖ": "_k",
    "ₗ": "_l",
    "ₘ": "_m",
    "ₙ": "_n",
    "ₒ": "_o",
    "ₚ": "_p",
    "ₛ": "_s",
    "ₜ": "_t",
    "ₓ": "_x",
    # Exposants Unicode
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "^+",
    "⁻": "^-",
    "⁼": "^=",
    "⁽": "^(",
    "⁾": "^)",
    "ⁱ": "^i",
    "ⁿ": "^n",
}

_MULTICHAR_UNICODE_TO_LATEX: dict[str, str] = {
    "\u0338=": r"\neq",
    "=\u0338": r"\neq",
    "<\u0338": r"\not<",
    ">\u0338": r"\not>",
    "\u0338∈": r"\notin",
    "∈\u0338": r"\notin",
}
_MULTICHAR_REGEX_TO_LATEX: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\u0338\s*="), r"\neq"),
    (re.compile(r"=\s*\u0338"), r"\neq"),
    (re.compile(r"\u0338\s*∈"), r"\notin"),
    (re.compile(r"∈\s*\u0338"), r"\notin"),
    (re.compile(r"<\s*\u0338"), r"\not<"),
    (re.compile(r">\s*\u0338"), r"\not>"),
)

MATH_SYMBOLS = (
    "Ωωαβγδεζηθικλμνξοπρστυφχψ"
    "∈∉⊂⊆⊄⊃⊇⊈⊉∪∩⇒⇔⟹⟺≤≥≦≧≠≈≃≡±∓×÷∞∑∏∫∬∭∮√∛∜"
    "∅∀∃∄∂∇∆∧∨¬∝∥⊥ℕℤℚℝℂℙ"
)
OPERATORS = "=+*/<>⇒⇔≤≥≠≈≃≡∈⊂⊆⊃⊇∪∩∧∨±∓×÷"
_MATH_FUNCTION_NAMES = (
    "arccos",
    "arcsin",
    "arctan",
    "arg",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "deg",
    "det",
    "dim",
    "exp",
    "gcd",
    "hom",
    "inf",
    "ker",
    "lg",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "max",
    "min",
    "Pr",
    "rank",
    "sec",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
    "tr",
)
_MATH_BUILTIN_FUNCTIONS = {
    "arccos",
    "arcsin",
    "arctan",
    "arg",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "deg",
    "det",
    "exp",
    "gcd",
    "inf",
    "lg",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "max",
    "min",
    "sec",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
}
_KNOWN_LATEX_COMMANDS = (
    "neq",
    "approx",
    "simeq",
    "cong",
    "equiv",
    "leq",
    "geq",
    "ll",
    "gg",
    "propto",
    "parallel",
    "perp",
    "in",
    "notin",
    "ni",
    "subset",
    "subseteq",
    "nsubset",
    "supset",
    "supseteq",
    "nsubseteq",
    "nsupseteq",
    "cup",
    "cap",
    "setminus",
    "emptyset",
    "forall",
    "exists",
    "nexists",
    "therefore",
    "because",
    "pm",
    "mp",
    "times",
    "cdot",
    "div",
    "sum",
    "prod",
    "int",
    "iint",
    "iiint",
    "oint",
    "sqrt",
    "frac",
    "partial",
    "nabla",
    "lfloor",
    "rfloor",
    "lceil",
    "rceil",
    "infty",
    "circ",
    "mathbb",
    "mathbf",
    "mathrm",
    "mathcal",
    "left",
    "right",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Phi",
    "Psi",
    "Omega",
    "sim",
    "to",
    "rightarrow",
    "leftarrow",
    "leftrightarrow",
    "longrightarrow",
    "Longrightarrow",
    "Longleftrightarrow",
    "mapsto",
    "Rightarrow",
    "Leftrightarrow",
    *_MATH_FUNCTION_NAMES,
)
_LATEX_COMMAND_RE = re.compile(
    r"\\(?:" + "|".join(sorted(map(re.escape, _KNOWN_LATEX_COMMANDS), key=len, reverse=True)) + r")(?![A-Za-z])"
)
_ANY_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_EXISTING_MATH_RE = re.compile(r"\$\$[^$]+\$\$|\$[^$]+\$", re.DOTALL)
_INDEX_PATTERN_RE = re.compile(r"\b[A-Za-zÀ-ÿ](?:_\{?[A-Za-z0-9+\-]+\}?|\^\{?[A-Za-z0-9+\-]+\}?)")
_SUPER_SUB_PAREN_RE = re.compile(
    r"(?:\([^()]*\)|[0-9]+)(?:[_^]\{[^{}]+\}|[_^][0-9A-Za-z])"
)
_FUNCTION_NAME_RE = "|".join(sorted(map(re.escape, _MATH_FUNCTION_NAMES), key=len, reverse=True))
_INLINE_MATH_TERM = (
    r"(?:"
    rf"\\(?:{_FUNCTION_NAME_RE})\s*\([^()]*\)"
    rf"|\\(?:{_FUNCTION_NAME_RE})\s+[A-Za-z0-9_{{}}^\\]+"
    r"|\\[A-Za-z]+(?:_\{?[A-Za-z0-9+\-]+\}?|\^\{?[A-Za-z0-9+\-]+\}?)*"
    r"|[A-Za-z](?:_\{?[A-Za-z0-9+\-]+\}?|\^\{?[A-Za-z0-9+\-]+\}?)*"
    r"|[0-9]+(?:\s*/\s*(?:[A-Za-z][A-Za-z0-9_{}]*|[0-9]+))?"
    r"|\([^()]*\)"
    r")"
)
_INLINE_MATH_EXPR = rf"{_INLINE_MATH_TERM}(?:\s*(?:[+\-*/]|\\cdot)\s*{_INLINE_MATH_TERM})*"
_INLINE_RELATION_RE = re.compile(
    rf"{_INLINE_MATH_EXPR}\s*(?:\\sim|\\approx|\\simeq|\\equiv|\\(?:long)?rightarrow|\\to|=|\\leq|\\geq|\\neq|\\in|\\notin|\\subset(?:eq)?|\\supset(?:eq)?)\s*{_INLINE_MATH_EXPR}"
)
_PROSE_HINT_RE = re.compile(
    r"\b(soit|suite|fonction|définie|definie|pour|quand|alors|avec|comme|"
    r"converge|diverge|terme|montre|donc|paragraphe|écrire|ecrire|revient|"
    r"encore|équivalent|equivalent|choisi|simplifier|changer|ordre|principal)\b",
    re.I,
)


def normalize_unicode_math(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for pattern, replacement in _MULTICHAR_REGEX_TO_LATEX:
        text = pattern.sub(lambda _match, value=replacement: f" {value} ", text)
    for pattern, replacement in _MULTICHAR_UNICODE_TO_LATEX.items():
        text = text.replace(pattern, f" {replacement} " if replacement.startswith("\\") else replacement)

    parts: list[str] = []
    for char in text:
        replacement = _UNICODE_TO_LATEX.get(char)
        if replacement is None:
            parts.append(char)
            continue
        if _is_latex_command(replacement):
            parts.append(f" {replacement} ")
        else:
            parts.append(replacement)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def normalize_math_text(text: str) -> str:
    raw = text
    text = normalize_unicode_math(text)
    text = text.replace("\u00a0", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = _repair_orphan_prose_prefix(text)
    text = _repair_glued_prose_math(text)
    text = _repair_standalone_scripts(text)
    text = _repair_sequence_greek_subscripts(text)
    text = _repair_latexish_artifacts(text)
    text = _repair_split_index_increments(text)
    text = _repair_script_runs(text)
    text = _repair_ascii_arrows(text)
    text = _repair_common_function_names(text)
    text = _repair_latexish_keywords(text)
    text = re.sub(r"(?<=[A-Za-z0-9}\)])\s*~\s*(?=[A-Za-z0-9({\\])", r" \\sim ", text)
    text = re.sub(r"(?:-\s*){2,}\s*\\rightarrow", r"\\longrightarrow", text)
    text = _repair_roots(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(rf"([{re.escape(MATH_SYMBOLS)}])([A-Za-zÀ-ÿ])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-zÀ-ÿ])([{re.escape(MATH_SYMBOLS)}])", r"\1 \2", text)
    text = re.sub(rf"\s*([{OPERATORS}])\s*", r" \1 ", text)
    text = _repair_script_spacing(text)
    text = re.sub(r"(?<=[0-9)\]}Ωωα-ω])\s*-\s*(?=[0-9({\[Ωωα-ω])", " - ", text)
    text = re.sub(r"\b([A-Za-z])\s*-\s*([A-Za-z])\b", r"\1 - \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:)\]}])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = _compact_script_braces(text)
    return text or raw.strip()


def looks_like_formula(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 260:
        return False
    if re.match(r"^\$\$?.+\$\$?$", stripped, re.DOTALL):
        return True
    unicode_symbol_hits = sum(1 for char in unicodedata.normalize("NFC", stripped) if char in _UNICODE_TO_LATEX)
    normalized = normalize_unicode_math(stripped)
    normalized = _repair_standalone_scripts(normalized)
    latex_commands = len(_LATEX_COMMAND_RE.findall(normalized))
    math_chars = len(re.findall(r"[=+*/<>|{}\[\]^_]", normalized)) + latex_commands
    latin_words = len(re.findall(r"\b[A-Za-zÀ-ÿ]{3,}\b", stripped))
    ratio = math_chars / max(len(normalized), 1)
    has_index_pattern = bool(_INDEX_PATTERN_RE.search(normalized))
    compact = len(normalized.split()) <= 6 and not _PROSE_HINT_RE.search(normalized)
    if _looks_like_mixed_prose_math(stripped, normalized):
        return False
    if _is_fraction_bar(normalized):
        return True
    if _is_arrow_only(normalized):
        return True
    if re.search(r"(?<=[A-Za-z0-9)\]}])\s*(?:=|<|>|\\leq|\\geq|\\neq|\\approx|\\sim)\s*(?=[A-Za-z0-9({\\])", normalized) and latin_words <= 2:
        return True
    if re.search(r"(?<=[A-Za-z0-9)\]}])\s*(?:[+\-*/])\s*(?=[A-Za-z0-9({\\])", normalized) and latin_words <= 2:
        return True
    if unicode_symbol_hits >= 2 and latin_words <= 6:
        return True
    if latex_commands >= 2 and latin_words <= 6:
        return True
    if has_index_pattern and compact:
        return True
    if "\\sim" in normalized and latin_words <= 6:
        return True
    return ratio >= 0.16 and latin_words <= 4


def normalize_math_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    normalized: list[DocumentBlock] = []
    for block in blocks:
        if block.type not in {"paragraph", "formula", "heading"}:
            normalized.append(block)
            continue
        if block.type == "formula" and block.metadata.get("source") == "math_zone_detector":
            metadata = dict(block.metadata)
            source_text = block.text or block.latex or ""
            normalized_text = normalize_math_text(normalize_unicode_math(source_text))

            block.text = normalized_text or source_text
            block.latex = _finalize_formula_latex(normalized_text or source_text)
            block.metadata = {
                **metadata,
                "render_mode": metadata.get("render_mode", "pdf_crop"),
                "raw_text": metadata.get("raw_text", source_text),
            }
            normalized.append(block)
            continue
        source_text = block.text or block.latex or ""
        raw_text = normalize_unicode_math(source_text)
        text = normalize_math_text(raw_text)
        metadata = dict(block.metadata)
        if source_text != text:
            metadata.setdefault("raw_text", source_text)

        mode = metadata.get("formula_mode")
        if block.type == "paragraph" and mode in {"inline", "ambiguous"} and _looks_like_standalone_display_math_piece(block, text):
            latex = _finalize_formula_latex(text)
            normalized.append(
                DocumentBlock(
                    type="formula",
                    text=_ensure_formula_delimiters(latex),
                    latex=latex,
                    page=block.page,
                    bbox=block.bbox,
                    confidence=min(block.confidence, 0.78),
                    metadata={
                        **metadata,
                        "source": metadata.get("source", "standalone_math_fragment"),
                        "formula_mode": "display",
                        "render_mode": "pdf_crop",
                        "preserve_bbox": True,
                    },
                )
            )
            continue

        if block.type == "paragraph" and mode in {"inline", "ambiguous"}:
            block.text = _wrap_inline_latex(text)
            block.metadata = metadata
            normalized.append(block)
            continue

        if block.type == "paragraph" and looks_like_formula(text):
            latex = _finalize_formula_latex(text)
            formula_text = _ensure_formula_delimiters(latex)
            normalized.append(
                DocumentBlock(
                    type="formula",
                    text=formula_text,
                    latex=latex,
                    page=block.page,
                    bbox=block.bbox,
                    confidence=block.confidence,
                    metadata=metadata,
                )
            )
            continue

        if block.type in {"paragraph", "heading"}:
            text = _wrap_inline_latex(text)

        block.text = text
        if block.type == "formula":
            block.latex = _finalize_formula_latex(block.latex or text)
            block.text = _ensure_formula_delimiters(block.latex)
        block.metadata = metadata
        normalized.append(block)
    return _cleanup_orphan_math_fragments(_merge_split_math_blocks(normalized))


def _looks_like_standalone_display_math_piece(block: DocumentBlock, text: str) -> bool:
    metadata = block.metadata or {}
    if metadata.get("is_metadata"):
        return False
    if block.bbox is None:
        return False

    raw_type = str(metadata.get("raw_block_type") or "")
    if raw_type not in {"line_with_inline_math", "ambiguous_math_line"} and not _uses_math_font(metadata):
        return False

    stripped = _strip_formula_delimiters(text or "").strip()
    if not stripped or len(stripped) > 72:
        return False
    if block.bbox.width > 180.0 or block.bbox.x0 < 110.0:
        return False

    prose_words = len(re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", re.sub(r"\\[A-Za-z]+", " ", stripped)))
    if prose_words > 1:
        return False

    if _uses_math_font(metadata) and re.fullmatch(r"[A-Za-z0-9∑Σ]+", stripped):
        return True
    return looks_like_formula(stripped)


def _uses_math_font(metadata: dict) -> bool:
    font = str(metadata.get("font_name") or metadata.get("font") or "").casefold()
    return any(marker in font for marker in ("math", "cmmi", "cmsy", "cmex", "stix"))


def _repair_standalone_scripts(text: str) -> str:
    """PyMuPDF can emit a lower-limit line as ``_{n}_{\\rightarrow}...``
    with no base. That is invalid LaTeX; read it as plain math tokens."""
    text = re.sub(
        r"^\s*(?:_\{[^{}]+\}\s*)+",
        lambda match: " ".join(re.findall(r"_\{([^{}]+)\}", match.group(0))) + " ",
        text,
    )
    return re.sub(
        r"(?<![A-Za-z0-9)\]}])_\{([^{}]+)\}",
        lambda match: f" {match.group(1)} ",
        text,
    )


def _repair_sequence_greek_subscripts(text: str) -> str:
    return re.sub(r"\\(epsilon|varepsilon)\s+([A-Za-z0-9])\b", r"\\\1_{\2}", text)


def _repair_split_index_increments(text: str) -> str:
    text = re.sub(r"([A-Za-z])_\{([^{}]+)\}_\{\+([^{}]+)\}", r"\1_{\2+\3}", text)
    text = re.sub(r"([A-Za-z])_\{([^{}]+)\}_\{-([^{}]+)\}", r"\1_{\2-\3}", text)
    return text


def _compact_script_braces(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group(2)
        if "\\" in body:
            compact = re.sub(r"\s+", " ", body).strip()
            return f"{match.group(1)}{{{compact}}}"
        compact = re.sub(r"\s+", "", body)
        return f"{match.group(1)}{{{compact}}}"

    return re.sub(
        r"([_^])\{([^{}]+)\}",
        repl,
        text,
    )


def _repair_script_runs(text: str) -> str:
    """Unicode exponent/subscript runs are often emitted as ``x^-^1`` or
    ``a_1_2``. Collapse contiguous same-kind scripts into one braced script."""

    def collapse(kind: str, source: str) -> str:
        pattern = re.compile(
            rf"(?P<base>(?:\\[A-Za-z]+|[A-Za-z0-9)\]}}]))"
            rf"(?P<run>(?:\{kind}(?:\{{[^{{}}]+\}}|[+\-=A-Za-z0-9])){{2,}})"
        )

        def repl(match: re.Match[str]) -> str:
            pieces = re.findall(rf"\{kind}(?:\{{([^{{}}]+)\}}|([+\-=A-Za-z0-9]))", match.group("run"))
            body = "".join(left or right for left, right in pieces)
            return f"{match.group('base')}{kind}{{{body}}}"

        return pattern.sub(repl, source)

    text = collapse("^", text)
    text = collapse("_", text)
    text = re.sub(r"(?P<base>(?:\\[A-Za-z]+|[A-Za-z0-9)\]}]))\^(?P<sign>[+\-=])(?P<body>[A-Za-z0-9]+)", r"\g<base>^{\g<sign>\g<body>}", text)
    text = re.sub(r"(?P<base>(?:\\[A-Za-z]+|[A-Za-z0-9)\]}]))_(?P<sign>[+\-=])(?P<body>[A-Za-z0-9]+)", r"\g<base>_{\g<sign>\g<body>}", text)
    return text


def _repair_script_spacing(text: str) -> str:
    text = re.sub(r"\s+([_^])\s*", r"\1", text)
    text = re.sub(r"([_^])\s+\{", r"\1{", text)
    return text


def _repair_ascii_arrows(text: str) -> str:
    text = re.sub(r"(?<![<\-])--+>", r" \\longrightarrow ", text)
    text = re.sub(r"(?<![<\-])-+>", r" \\rightarrow ", text)
    text = re.sub(r"<-+(?!>)", r" \\leftarrow ", text)
    text = re.sub(r"<=>", r" \\Leftrightarrow ", text)
    text = re.sub(r"=>", r" \\Rightarrow ", text)
    return text


def _repair_common_function_names(text: str) -> str:
    function_names = _FUNCTION_NAME_RE
    text = re.sub(rf"(?<!\\)\b({function_names})\s*(?=\()", lambda m: _format_math_function(m.group(1)), text)
    text = re.sub(rf"(?<!\\)\b({function_names})\s*(?=_[{{A-Za-z0-9+\-])", lambda m: _format_math_function(m.group(1)), text)
    text = re.sub(
        r"(?<!\\)\b(ln|log|exp|sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh)\s+(?=[A-Za-z0-9\\(])",
        lambda m: f"{_format_math_function(m.group(1))} ",
        text,
    )
    return text


def _format_math_function(name: str) -> str:
    return f"\\{name}" if name in _MATH_BUILTIN_FUNCTIONS else rf"\mathrm{{{name}}}"


def _repair_latexish_keywords(text: str) -> str:
    replacements = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "pi": r"\pi",
        "sigma": r"\sigma",
        "omega": r"\omega",
        "infty": r"\infty",
    }

    def repl(match: re.Match[str]) -> str:
        return replacements[match.group(1)]

    # Only repair text that already looks like a formula; this avoids turning
    # prose words such as "alpha" into math in ordinary paragraphs.
    if not re.search(r"[_^=+\-*/{}]|\\|[0-9]\s*(?:,|\.|$)", text):
        return text
    return re.sub(r"(?<!\\)\b(" + "|".join(replacements) + r")\b", repl, text)


def _repair_roots(text: str) -> str:
    text = re.sub(r"\bsqrt\s*\(([^()]+)\)", r"\\sqrt{\1}", text)
    text = re.sub(r"\bsqrt\s+([A-Za-z0-9_{}^\\]+)", r"\\sqrt{\1}", text)
    text = re.sub(r"\\sqrt\[(\d+)\]\s*\(([^()]+)\)", r"\\sqrt[\1]{\2}", text)
    text = re.sub(r"\\sqrt\[(\d+)\]\s*([A-Za-z0-9_{}^\\]+)", r"\\sqrt[\1]{\2}", text)
    text = re.sub(r"\\sqrt\s*\(([^()]+)\)", r"\\sqrt{\1}", text)
    text = re.sub(r"\\sqrt\s+([A-Za-z0-9_{}^\\]+)", r"\\sqrt{\1}", text)
    return text


def _repair_latexish_artifacts(text: str) -> str:
    text = re.sub(r"e\^\{\s*-\s*\\?\s*\}\s*\^\{([^{}]+)\}", r"e^{-\1}", text)
    text = re.sub(r"e\^\{\s*-\s*\\?\s*\}\s*\^([A-Za-z0-9])", r"e^{-\1}", text)
    text = re.sub(r"e\^\{\s*-\s*\\?\s*\}\s*([A-Za-z0-9])", r"e^{-\1}", text)
    text = re.sub(r"e\^\{\s*-\s*\\lambda\s*\^\{?([A-Za-z0-9]+)\}?", r"e^{-\\lambda^{\1}}", text)
    return _balance_latex_braces(text)


def _balance_latex_braces(text: str) -> str:
    if _PROSE_HINT_RE.search(text):
        return text
    opened = text.count("{")
    closed = text.count("}")
    if opened > closed and opened - closed <= 3:
        return text + ("}" * (opened - closed))
    return text


def _repair_orphan_prose_prefix(text: str) -> str:
    return re.sub(
        r"^\s*[.·]\s*[A-Za-z]\s+(?=(?:Comme|Donc|Ainsi|Alors|Or|En particulier)\b)",
        "",
        text,
        flags=re.I,
    )


def _repair_glued_prose_math(text: str) -> str:
    text = re.sub(r"\b(écrire|ecrire)(?=[A-Za-z]_\{?[A-Za-z0-9])", r"\1 ", text, flags=re.I)
    text = re.sub(
        r"(_\{?[A-Za-z0-9]+\}?)(?=(?:revient|écrire|ecrire|est|sont|avec|ou|et))",
        r"\1 ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(revient)(?=à|a\b)", r"\1 ", text, flags=re.I)
    text = re.sub(r"\b(à|a)(?=écrire|ecrire\b)", r"\1 ", text, flags=re.I)
    return text


def _looks_like_mixed_prose_math(source: str, normalized: str) -> bool:
    if not _PROSE_HINT_RE.search(normalized):
        return False
    prose_words = [
        word
        for word in re.findall(r"\b[A-Za-zÀ-ÿ]{3,}\b", source)
        if not re.fullmatch(r"[A-Za-z]_[A-Za-z0-9]+", word)
    ]
    if len(prose_words) >= 2:
        return True
    return bool(re.search(r"[A-Za-zÀ-ÿ]{4,}.*(?:\\sim|\\rightarrow|_|=)|(?:\\sim|\\rightarrow|_|=).*[A-Za-zÀ-ÿ]{4,}", normalized))


def _merge_split_math_blocks(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    result: list[DocumentBlock] = []
    i = 0
    while i < len(blocks):
        if blocks[i].type == "formula" and blocks[i].metadata.get("source") == "math_zone_detector":
            result.append(blocks[i])
            i += 1
            continue

        merged = _try_merge_equivalence_definition(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_leading_limit_arrow(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_split_limit_expression(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_split_unit_fraction(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_stacked_fraction_bar(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_visual_fraction(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_math_continuation(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        merged = _try_merge_stacked_fraction(blocks, i)
        if merged is not None:
            block, next_i = merged
            result.append(block)
            i = next_i
            continue

        result.append(blocks[i])
        i += 1
    return result


def _cleanup_orphan_math_fragments(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    cleaned: list[DocumentBlock] = []
    for index, block in enumerate(blocks):
        if block.type == "paragraph" and _is_useless_formula_fragment(block.text):
            continue
        if block.type != "formula":
            cleaned.append(block)
            continue
        if block.metadata.get("source") == "math_zone_detector":
            cleaned.append(block)
            continue
        latex = _clean_math_fragment(block)
        if _is_useless_formula_fragment(latex):
            continue
        if _is_orphan_numeric_fragment(blocks, index, latex):
            continue
        cleaned.append(block)
    return cleaned


def _is_useless_formula_fragment(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", _strip_formula_delimiters(text))
    if not cleaned:
        return True
    if _is_truncated_latex_command_fragment(cleaned):
        return True
    if re.search(r"[A-Za-z0-9]|\\[A-Za-z]+", cleaned):
        return False
    return bool(re.fullmatch(r"[()[\]{}.,;:+\-*/=\\|]+", cleaned))


def _is_truncated_latex_command_fragment(text: str) -> bool:
    return text in {r"\tex", r"\text"}


def _is_orphan_numeric_fragment(blocks: list[DocumentBlock], index: int, text: str) -> bool:
    cleaned = _strip_formula_delimiters(text).strip()
    if len(cleaned) > 20:
        return False
    if re.search(r"[A-Za-z\\]", cleaned):
        return False
    if not re.fullmatch(r"[0-9\s+\-*/().,=]+", cleaned):
        return False
    if not re.search(r"[+\-*/=]", cleaned):
        return False

    for neighbor_index in (index - 2, index - 1, index + 1, index + 2):
        if neighbor_index < 0 or neighbor_index >= len(blocks):
            continue
        neighbor = blocks[neighbor_index]
        if neighbor.type != "formula":
            continue
        neighbor_text = _plain_block_text(neighbor)
        if not re.search(r"[A-Za-z\\]", neighbor_text):
            continue
        if _same_page(blocks[index], neighbor) and _blocks_are_visually_close(blocks[index], neighbor, factor=5.0):
            return True
    return False


def _try_merge_equivalence_definition(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    block = blocks[index]
    if block.type != "formula":
        return None

    relation = _clean_math_fragment(block)
    if not _contains_equivalence(relation):
        return None

    j = index + 1
    if j < len(blocks) and _same_page(block, blocks[j]) and blocks[j].type == "formula":
        continuation = _clean_math_fragment(blocks[j])
        if _is_limit_parenthetical(continuation):
            relation = f"{relation} {continuation}".strip()
            j += 1

    if j >= len(blocks) or not _same_page(block, blocks[j]) or not _is_si_connector(blocks[j]):
        return None

    fraction = _parse_stacked_fraction(blocks, j + 1)
    if fraction is None:
        return None

    fraction_latex, next_i = fraction
    latex = f"{relation} \\quad \\mathrm{{si}} \\quad {fraction_latex}"
    return _build_formula_block(blocks[index:next_i], latex), next_i


def _try_merge_split_limit_expression(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index >= len(blocks) or blocks[index].type != "formula":
        return None

    expression = _clean_math_fragment(blocks[index])
    prefix = _strip_trailing_arrow(expression)
    if prefix is None or not _has_equation_context(prefix):
        return None

    parsed = _parse_limit_and_target(blocks, index + 1)
    if parsed is None:
        return None

    limit, target, next_i = parsed
    latex = f"{prefix}\\rightarrow_{{{limit}}}{target}"
    return _build_formula_block(blocks[index:next_i], latex), next_i


def _try_merge_leading_limit_arrow(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index + 2 >= len(blocks) or blocks[index].type != "formula" or blocks[index + 1].type != "formula":
        return None
    if not _same_page(blocks[index], blocks[index + 1]):
        return None

    arrow_prefix = _strip_trailing_arrow(_clean_math_fragment(blocks[index]))
    expression = _clean_math_fragment(blocks[index + 1])
    if arrow_prefix is None or not _is_simple_math_atom(arrow_prefix) or not _has_equation_context(expression):
        return None

    parsed = _parse_limit_and_target(blocks, index + 2)
    if parsed is None:
        return None

    limit, target, next_i = parsed
    separator = " " if expression.endswith(",") else ", "
    latex = f"{expression}{separator}{arrow_prefix}\\rightarrow_{{{limit}}}{target}"
    return _build_formula_block(blocks[index:next_i], latex), next_i


def _try_merge_split_unit_fraction(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index + 1 >= len(blocks) or blocks[index].type != "formula" or blocks[index + 1].type != "formula":
        return None
    if not _same_page(blocks[index], blocks[index + 1]):
        return None

    left = _clean_math_fragment(blocks[index])
    right = _clean_math_fragment(blocks[index + 1])
    if not re.search(r"(?:\\cdot|\*)\s*1\s*$", left):
        return None

    match = re.match(r"^(?P<den>[A-Za-z0-9_{}\\]+)\s*=\s*(?P<rhs>.+)$", right)
    if not match:
        return None

    denominator = match.group("den").strip()
    rhs = match.group("rhs").strip()
    if not denominator or _PROSE_HINT_RE.search(rhs):
        return None

    prefix = re.sub(r"(?:\\cdot|\*)\s*1\s*$", "", left).rstrip()
    latex = rf"{prefix}\cdot \frac{{1}}{{{denominator}}} = {rhs}"
    return _build_formula_block(blocks[index : index + 2], latex), index + 2


def _try_merge_stacked_fraction_bar(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index + 2 >= len(blocks) or not _is_math_fragment_candidate(blocks[index]):
        return None
    if not _same_page(blocks[index], blocks[index + 1]) or not _same_page(blocks[index], blocks[index + 2]):
        return None
    if not _is_fraction_bar(_clean_math_fragment(blocks[index + 1])):
        return None
    if not _is_math_fragment_candidate(blocks[index + 2]):
        return None

    numerator = _clean_math_fragment(blocks[index])
    denominator = _clean_math_fragment(blocks[index + 2])
    if not _is_fraction_component(numerator) or not _is_fraction_component(denominator):
        return None
    if not _blocks_are_visually_close(blocks[index], blocks[index + 1], factor=1.5):
        return None
    if not _blocks_are_visually_close(blocks[index + 1], blocks[index + 2], factor=1.5):
        return None

    latex = rf"\frac{{{numerator}}}{{{denominator}}}"
    return _build_formula_block(blocks[index : index + 3], latex), index + 3


def _try_merge_visual_fraction(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index + 1 >= len(blocks) or blocks[index].type != "formula" or blocks[index + 1].type != "formula":
        return None
    if not _same_page(blocks[index], blocks[index + 1]):
        return None
    if not _blocks_are_visually_close(blocks[index], blocks[index + 1], factor=0.8):
        return None

    numerator = _clean_math_fragment(blocks[index])
    denominator = _clean_math_fragment(blocks[index + 1])
    if not _is_fraction_component(numerator) or not _is_fraction_component(denominator):
        return None
    if _has_relation_operator(numerator) or _has_relation_operator(denominator):
        return None
    if not _is_centered_over(blocks[index], blocks[index + 1]):
        return None

    latex = rf"\frac{{{numerator}}}{{{denominator}}}"
    return _build_formula_block(blocks[index : index + 2], latex), index + 2


def _try_merge_math_continuation(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    if index + 1 >= len(blocks) or blocks[index].type != "formula" or not _is_math_fragment_candidate(blocks[index + 1]):
        return None
    if not _same_page(blocks[index], blocks[index + 1]):
        return None
    if not _blocks_are_visually_close(blocks[index], blocks[index + 1], factor=1.8):
        return None

    left = _clean_math_fragment(blocks[index])
    right = _clean_math_fragment(blocks[index + 1])
    if not left or not right:
        return None
    if _is_fraction_bar(left) or _is_fraction_bar(right):
        return None
    if not (_math_line_needs_continuation(left) or _math_line_continues_previous(right)):
        return None

    parts = [left, right]
    next_i = index + 2
    while next_i < len(blocks) and _is_math_fragment_candidate(blocks[next_i]) and _same_page(blocks[index], blocks[next_i]):
        if not _blocks_are_visually_close(blocks[next_i - 1], blocks[next_i], factor=1.8):
            break
        fragment = _clean_math_fragment(blocks[next_i])
        if not fragment or not (_math_line_needs_continuation(parts[-1]) or _math_line_continues_previous(fragment)):
            break
        parts.append(fragment)
        next_i += 1

    return _build_formula_block(blocks[index:next_i], " ".join(parts)), next_i


def _try_merge_stacked_fraction(
    blocks: list[DocumentBlock],
    index: int,
) -> tuple[DocumentBlock, int] | None:
    parsed = _parse_stacked_fraction(blocks, index)
    if parsed is None:
        return None
    latex, next_i = parsed
    return _build_formula_block(blocks[index:next_i], latex), next_i


def _parse_stacked_fraction(blocks: list[DocumentBlock], index: int) -> tuple[str, int] | None:
    if index >= len(blocks) or blocks[index].type != "formula":
        return None

    numerator = _clean_math_fragment(blocks[index])
    if not _is_simple_math_atom(numerator):
        return None

    denominator: str | None = None
    limit: str | None = None
    target: str | None = None
    saw_arrow = False
    next_i = index + 1

    while next_i < len(blocks) and next_i - index <= 7:
        current = blocks[next_i]
        if not _same_page(blocks[index], current) or not _is_math_fragment_candidate(current):
            break

        fragment = _clean_math_fragment(current)
        if _is_arrow_only(fragment):
            saw_arrow = True
        else:
            limit_target = _split_limit_target(fragment)
            if limit_target is not None:
                limit, maybe_target = limit_target
                saw_arrow = True
                if maybe_target:
                    target = maybe_target
            elif _is_numeric_target(fragment):
                target = fragment
            elif _is_simple_math_atom(fragment):
                denominator = fragment
            else:
                break
        next_i += 1

    if denominator is None or target is None or not saw_arrow:
        return None

    arrow = r"\rightarrow"
    if limit:
        arrow += f"_{{{limit}}}"
    latex = rf"\frac{{{numerator}}}{{{denominator}}}{arrow}{target or ''}"
    return latex, next_i


def _parse_limit_and_target(blocks: list[DocumentBlock], index: int) -> tuple[str, str, int] | None:
    if index >= len(blocks) or not _is_math_fragment_candidate(blocks[index]):
        return None

    first = _clean_math_fragment(blocks[index])
    split = _split_limit_target(first)
    if split is not None:
        limit, target = split
        if target is not None:
            return limit, target, index + 1
        if index + 1 < len(blocks) and _same_page(blocks[index], blocks[index + 1]):
            maybe_target = _clean_math_fragment(blocks[index + 1])
            if _is_numeric_target(maybe_target):
                return limit, maybe_target, index + 2
    return None


def _is_math_fragment_candidate(block: DocumentBlock) -> bool:
    if block.type == "formula":
        return True
    text = _plain_block_text(block)
    if len(text) > 45:
        return False
    normalized = normalize_math_text(text)
    return (
        _is_arrow_only(normalized)
        or _is_numeric_target(normalized)
        or _split_limit_target(normalized) is not None
        or _is_compact_math_row(normalized)
        or looks_like_formula(normalized)
    )


def _clean_math_fragment(block: DocumentBlock) -> str:
    text = block.latex if block.type == "formula" and block.latex else block.text
    text = _strip_formula_delimiters(text or "")
    text = normalize_math_text(text)
    text = _strip_formula_delimiters(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _plain_block_text(block: DocumentBlock) -> str:
    return (block.text or block.latex or "").strip()


def _same_page(left: DocumentBlock, right: DocumentBlock) -> bool:
    return left.page is None or right.page is None or left.page == right.page


def _contains_equivalence(text: str) -> bool:
    return bool(re.search(r"(?:\\sim|∼|(?<!\\)~)", text))


def _is_limit_parenthetical(text: str) -> bool:
    return bool(re.match(r"^\(?\s*[^()]*\\(?:long)?rightarrow[^()]*\\infty[^()]*\)?[,]?$", text))


def _is_si_connector(block: DocumentBlock) -> bool:
    return _plain_block_text(block).casefold().strip(" .,:;") == "si"


def _is_arrow_only(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", _strip_formula_delimiters(text))
    return bool(re.fullmatch(r"(?:[-–—]+)?\\(?:long)?rightarrow", cleaned))


def _is_fraction_bar(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", _strip_formula_delimiters(text))
    return bool(re.fullmatch(r"(?:[-–—_]|\\overline\{\}){3,}|\\?[-–—]{3,}", cleaned))


def _split_limit_target(text: str) -> tuple[str, str | None] | None:
    cleaned = _strip_formula_delimiters(text).strip(" ;")
    match = re.match(
        r"^(?P<limit>.+?\\(?:long)?rightarrow\s*\+?\s*\\infty)\s*(?P<target>[A-Za-z0-9]+[\.,]?)?$",
        cleaned,
    )
    if not match and cleaned.endswith(","):
        match = re.match(
            r"^(?P<limit>.+?\\(?:long)?rightarrow\s*\+?\s*\\infty),$",
            cleaned,
        )
    if not match:
        return None
    limit = re.sub(r"\s+", " ", match.group("limit")).strip()
    target = ""
    if "target" in match.groupdict():
        target = match.group("target") or ""
    target = target.strip() or None
    return limit, target


def _is_numeric_target(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]+[\.,]?", text.strip()))


def _is_compact_math_row(text: str) -> bool:
    cleaned = _strip_formula_delimiters(text).strip()
    if not cleaned or len(cleaned) > 90 or _PROSE_HINT_RE.search(cleaned):
        return False
    if not re.fullmatch(r"[\[\](){},;0-9A-Za-z+\-*/^_\\.\s]+", cleaned):
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    has_delimiter = any(char in cleaned for char in "()[]{}")
    return has_delimiter and len(tokens) >= 2


def _is_simple_math_atom(text: str) -> bool:
    cleaned = _strip_formula_delimiters(text).strip()
    if not cleaned or len(cleaned) > 40:
        return False
    if any(token in cleaned for token in ("\\rightarrow", "\\longrightarrow", "\\infty", "\\frac", "\\sim", "=", ",")):
        return False
    return not _PROSE_HINT_RE.search(cleaned)


def _is_fraction_component(text: str) -> bool:
    cleaned = _strip_formula_delimiters(text).strip()
    if not cleaned or len(cleaned) > 140:
        return False
    if _PROSE_HINT_RE.search(cleaned):
        return False
    if cleaned.count("{") != cleaned.count("}"):
        return False
    return bool(
        re.search(r"[A-Za-z0-9\\]", cleaned)
        or any(symbol in cleaned for symbol in ("+", "-", "=", "\\sum", "\\int", "\\sqrt"))
    )


def _has_relation_operator(text: str) -> bool:
    return bool(
        re.search(
            r"(?:=|<|>|\\leq|\\geq|\\neq|\\sim|\\approx|\\simeq|\\equiv|\\(?:long)?rightarrow|\\to|\\in|\\subset)",
            text,
        )
    )


def _math_line_needs_continuation(text: str) -> bool:
    stripped = _strip_formula_delimiters(text).strip()
    if not stripped:
        return False
    if re.search(r"(?:=|[+\-*/]|\\cdot|\\times|\\frac\{[^{}]*\}\{?|\\sqrt\{?|\\left[({\[]?)\s*$", stripped):
        return True
    if stripped.count("{") > stripped.count("}") or stripped.count("(") > stripped.count(")"):
        return True
    return False


def _math_line_continues_previous(text: str) -> bool:
    stripped = _strip_formula_delimiters(text).strip()
    if not stripped:
        return False
    return bool(re.match(r"^(?:=|[+\-*/]|\\cdot|\\times|\\right|\\leq|\\geq|\\sim|\\approx|\\to|\\rightarrow)", stripped))


def _blocks_are_visually_close(left: DocumentBlock, right: DocumentBlock, factor: float = 2.0) -> bool:
    if left.bbox is None or right.bbox is None:
        return True
    if right.bbox.y0 < left.bbox.y0:
        return False
    gap = max(0.0, right.bbox.y0 - left.bbox.y1)
    average_height = max(1.0, (left.bbox.height + right.bbox.height) / 2.0)
    return gap <= max(6.0, average_height * factor)


def _is_centered_over(upper: DocumentBlock, lower: DocumentBlock) -> bool:
    if upper.bbox is None or lower.bbox is None:
        return False
    center_delta = abs(upper.bbox.center_x - lower.bbox.center_x)
    wider = max(upper.bbox.width, lower.bbox.width, 1.0)
    overlap = min(upper.bbox.x1, lower.bbox.x1) - max(upper.bbox.x0, lower.bbox.x0)
    return center_delta <= wider * 0.25 or overlap > min(upper.bbox.width, lower.bbox.width) * 0.45


def _strip_trailing_arrow(text: str) -> str | None:
    match = re.match(r"^(?P<prefix>.+?)\\(?:long)?rightarrow\s*$", _strip_formula_delimiters(text).strip())
    if not match:
        return None
    prefix = re.sub(r"\s+", " ", match.group("prefix")).strip()
    return prefix or None


def _has_equation_context(text: str) -> bool:
    return "=" in text or "," in text


def _build_formula_block(source: list[DocumentBlock], latex: str) -> DocumentBlock:
    first = source[0]
    bbox = first.bbox
    for block in source[1:]:
        if bbox is not None and block.bbox is not None:
            bbox = bbox.union(block.bbox)
        elif bbox is None:
            bbox = block.bbox

    metadata = dict(first.metadata)
    metadata["merged_math_fragments"] = len(source)
    latex = _finalize_formula_latex(latex)
    return DocumentBlock(
        type="formula",
        text=_ensure_formula_delimiters(latex),
        latex=latex,
        page=first.page,
        bbox=bbox,
        confidence=min(block.confidence for block in source),
        metadata=metadata,
    )


_SIMPLE_FRACTION_ATOM = (
    r"(?:"
    r"\\[A-Za-z]+(?:\{[^{}]+\})?(?:_\{[^{}]+\}|_[A-Za-z0-9]+|\^\{[^{}]+\}|\^[A-Za-z0-9]+)*"
    r"|[A-Za-z0-9]+(?:_\{[^{}]+\}|_[A-Za-z0-9]+|\^\{[^{}]+\}|\^[A-Za-z0-9]+)*"
    r"|\([^()]+\)"
    r")"
)


def _finalize_formula_latex(text: str) -> str:
    latex = _strip_formula_delimiters(text or "")
    latex = normalize_math_text(latex)
    latex = _repair_operator_commands_for_mathtext(latex)
    latex = _repair_inline_fractions(latex)
    latex = _balance_latex_braces(latex)
    return latex.strip()


def _repair_operator_commands_for_mathtext(text: str) -> str:
    # Opérateurs non standards → \mathrm
    text = re.sub(
        r"\\(rank|tr|ker|dim|hom|Pr|card|pgcd|ppcm|lcm|vect|Vect|Im|Re|id|Id)\b",
        lambda m: rf"\mathrm{{{m.group(1)}}}",
        text,
    )
    # \vec{x} → \mathbf{x}  (non supporté par mathtext)
    text = re.sub(r"\\vec\{([^{}]+)\}", r"\\mathbf{\1}", text)
    # \overrightarrow{AB} → \stackrel{\rightarrow}{AB}
    text = re.sub(r"\\overrightarrow\{([^{}]+)\}", r"\\stackrel{\\rightarrow}{\1}", text)
    # \norm{x} → \|x\|   (commande custom courante en CPGE)
    text = re.sub(r"\\norm\{([^{}]+)\}", r"\\| \1 \\|", text)
    # \abs{x} → |x|
    text = re.sub(r"\\abs\{([^{}]+)\}", r"|\1|", text)
    # \floor{x} → \lfloor x \rfloor
    text = re.sub(r"\\floor\{([^{}]+)\}", r"\\lfloor \1 \\rfloor", text)
    # \ceil{x} → \lceil x \rceil
    text = re.sub(r"\\ceil\{([^{}]+)\}", r"\\lceil \1 \\rceil", text)
    # \operatorname{f} → \mathrm{f}
    text = re.sub(r"\\operatorname\{([^{}]+)\}", r"\\mathrm{\1}", text)
    # \widehat{x} → \hat{x}
    text = re.sub(r"\\widehat\{([^{}]+)\}", r"\\hat{\1}", text)
    # \widetilde{x} → \tilde{x}
    text = re.sub(r"\\widetilde\{([^{}]+)\}", r"\\tilde{\1}", text)
    # \boldsymbol{x} → \mathbf{x}
    text = re.sub(r"\\boldsymbol\{([^{}]+)\}", r"\\mathbf{\1}", text)
    return text


def _repair_inline_fractions(text: str) -> str:
    if "/" not in text:
        return text

    function_atom = rf"\\(?:sin|cos|tan|log|ln|exp|arcsin|arccos|arctan|sinh|cosh|tanh)\s+{_SIMPLE_FRACTION_ATOM}"
    function_pattern = re.compile(
        rf"(?<![A-Za-z0-9}}])(?P<num>{function_atom})\s*/\s*(?P<den>{_SIMPLE_FRACTION_ATOM})(?![A-Za-z0-9{{])"
    )

    def frac(num: str, den: str) -> str:
        if num.startswith(r"\frac") or den.startswith(r"\frac"):
            return f"{num} / {den}"
        return rf"\frac{{{num.strip()}}}{{{den.strip()}}}"

    text = function_pattern.sub(lambda match: frac(match.group("num"), match.group("den")), text)

    pattern = re.compile(
        rf"(?<![A-Za-z0-9}}])(?P<num>{_SIMPLE_FRACTION_ATOM})\s*/\s*(?P<den>{_SIMPLE_FRACTION_ATOM})(?![A-Za-z0-9{{])"
    )

    def repl(match: re.Match[str]) -> str:
        numerator = match.group("num").strip()
        denominator = match.group("den").strip()
        return frac(numerator, denominator)

    return pattern.sub(repl, text)


def _wrap_inline_latex(text: str) -> str:
    """Wrap bare LaTeX commands and index patterns outside $...$ in $...$."""
    protected = [(m.start(), m.end()) for m in _EXISTING_MATH_RE.finditer(text)]

    def is_protected(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in protected)

    matches: list[tuple[int, int, str]] = []
    for m in _INLINE_RELATION_RE.finditer(text):
        if not is_protected(m.start(), m.end()):
            matches.append((m.start(), m.end(), m.group()))

    for pattern in (_ANY_LATEX_CMD_RE, _INDEX_PATTERN_RE, _SUPER_SUB_PAREN_RE):
        for m in pattern.finditer(text):
            if not is_protected(m.start(), m.end()):
                matches.append((m.start(), m.end(), m.group()))

    if not matches:
        return _repair_wrapped_inline_artifacts(text)

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    filtered: list[tuple[int, int, str]] = []
    last_end = 0
    for start, end, group in matches:
        if start >= last_end:
            filtered.append((start, end, group))
            last_end = end

    parts: list[str] = []
    last = 0
    for start, end, group in filtered:
        parts.append(text[last:start])
        parts.append(f"${group}$")
        last = end
    parts.append(text[last:])
    return _repair_wrapped_inline_artifacts("".join(parts))


def _repair_wrapped_inline_artifacts(text: str) -> str:
    text = re.sub(r"\brelatio\$n\s*\\sim\s*e\$st\b", r"relation $\\sim$ est", text)
    text = re.sub(r"\\\$(\\sqrt)\$\{([^{}]+)\}", r"$\1{\2}$", text)
    text = re.sub(r"^\s*en\s+(?=donc\b)", "", text, flags=re.I)
    return text


def _is_latex_command(value: str) -> bool:
    return value.startswith("\\") and len(value) > 1 and value[1].isalpha()


def _ensure_formula_delimiters(text: str) -> str:
    stripped = text.strip()
    if re.match(r"^\$\$?.+\$\$?$", stripped, re.DOTALL):
        return stripped
    return f"${stripped}$"


def _strip_formula_delimiters(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped[1:-1].strip()
    return stripped
