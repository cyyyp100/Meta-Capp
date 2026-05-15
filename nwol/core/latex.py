# core/latex.py — Rendu formules LaTeX → image Tkinter via matplotlib mathtext
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache

logger = logging.getLogger("LaTeX")

# Commandes non supportées par matplotlib mathtext → substituts compatibles
_COMPAT_SUBS: list[tuple[re.Pattern[str], str]] = [
    # \vec{x} → \mathbf{x}  (mathtext n'a pas \vec)
    (re.compile(r"\\vec\{([^{}]+)\}"), r"\\mathbf{\1}"),
    # \overrightarrow{AB} → \stackrel{\rightarrow}{AB}
    (re.compile(r"\\overrightarrow\{([^{}]+)\}"), r"\\stackrel{\\rightarrow}{\1}"),
    # \overleftarrow{AB} → \stackrel{\leftarrow}{AB}
    (re.compile(r"\\overleftarrow\{([^{}]+)\}"), r"\\stackrel{\\leftarrow}{\1}"),
    # \operatorname{f} → \mathrm{f}
    (re.compile(r"\\operatorname\{([^{}]+)\}"), r"\\mathrm{\1}"),
    # \widehat{x} → \hat{x}
    (re.compile(r"\\widehat\{([^{}]+)\}"), r"\\hat{\1}"),
    # \widetilde{x} → \tilde{x}
    (re.compile(r"\\widetilde\{([^{}]+)\}"), r"\\tilde{\1}"),
    # \not= → \neq
    (re.compile(r"\\not\s*="), r"\\neq"),
    # \not< → \not<  (garder tel quel, matplotlib le gère)
    # \ell → l  (si non supporté)
    # \, \; \: → espace fine (mathtext les ignore proprement)
    (re.compile(r"\\[,;:]"), " "),
    # \! → rien (espace négative, ignorée)
    (re.compile(r"\\!"), ""),
    # \quad \qquad → espace
    (re.compile(r"\\q?quad"), " \\; "),
    # \text{...} → \mathrm{...}  (plus robuste dans mathtext)
    (re.compile(r"\\text\{([^{}]*)\}"), r"\\mathrm{\1}"),
    # Environnements non supportés → supprimés (begin/end)
    (re.compile(r"\\begin\{[^}]+\}|\\end\{[^}]+\}"), ""),
    # \hline, \\ → ignorés (dans les matrices)
    (re.compile(r"\\hline|\\\\"), " "),
    # & (séparateur colonne) → espace
    (re.compile(r"(?<!\\)&"), " "),
    # Doubles backslashes restants
    (re.compile(r"\\\\"), " "),
]

# Détection d'un environnement matriciel
_MATRIX_ENV_RE = re.compile(r"\\begin\{(?:p?matrix|array|cases|bmatrix|vmatrix|Vmatrix)\}")
_TRUNCATED_COMMAND_RE = re.compile(r"\\(?:tex|text)\s*$")
_COMMAND_REQUIRING_ARGUMENT_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|mathrm|mathbf|mathbb|mathcal|mathsf|mathtt|operatorname|"
    r"hat|tilde|bar|overline|underline|vec|overrightarrow|overleftarrow|stackrel)\s*$"
)


def _prepare_latex(latex: str) -> str:
    """Prépare le LaTeX brut pour matplotlib mathtext."""
    s = latex.strip()
    # Retirer délimiteurs $...$ ou $$...$$
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    for pattern, replacement in _COMPAT_SUBS:
        s = pattern.sub(replacement, s)

    # Rééquilibrer les accolades si nécessaire
    opened = s.count("{")
    closed = s.count("}")
    if opened > closed:
        s += "}" * (opened - closed)
    elif closed > opened:
        s = "{" * (closed - opened) + s

    return s.strip()


def _has_unsupported_env(latex: str) -> bool:
    return bool(_MATRIX_ENV_RE.search(latex))


def _is_obviously_unrenderable_latex(latex: str) -> bool:
    s = (latex or "").strip()
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if not s:
        return True
    if _TRUNCATED_COMMAND_RE.fullmatch(s):
        return True
    return bool(_COMMAND_REQUIRING_ARGUMENT_RE.search(s))


@lru_cache(maxsize=512)
def render_formula(latex: str, display: bool = True, dpi: int = 180) -> bytes | None:
    """
    Génère une image PNG (bytes) d'une formule LaTeX via matplotlib mathtext.
    Utilise la police STIX pour supporter \\mathbb, \\mathcal, etc.
    Retourne None en cas d'échec.
    """
    if _is_obviously_unrenderable_latex(latex):
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Police STIX : supporte \mathbb{N}, \mathbb{R}, \mathcal{L}, etc.
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"

        # Environnements matriciels non supportés → fallback texte
        if _has_unsupported_env(latex):
            return _render_as_text(latex, dpi)

        prepared = _prepare_latex(latex)
        if not prepared:
            return None

        expr = f"${prepared}$"
        fontsize = 20 if display else 16

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, expr, fontsize=fontsize, color="#1A1A1A")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    transparent=True, pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        logger.warning("Rendu LaTeX échoué (%r) : %s — tentative simplifiée", latex[:60], exc)
        return _render_simplified(latex, dpi)


def _render_simplified(latex: str, dpi: int) -> bytes | None:
    """Deuxième tentative avec expression simplifiée (retire commandes inconnues)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["mathtext.fontset"] = "stix"

        # Supprimer toutes les commandes \xxx inconnues pour ne garder que la structure
        cleaned = re.sub(r"\\[a-zA-Z]+", "", _prepare_latex(latex))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None

        expr = f"${cleaned}$"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, expr, fontsize=15, color="#1A1A1A")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    transparent=True, pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.error("Rendu simplifié échoué : %s", exc)
        return None


def _render_as_text(latex: str, dpi: int) -> bytes | None:
    """Rendu d'une expression comme texte brut (fallback pour matrices)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Nettoyer pour affichage texte
        text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", latex)
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        text = re.sub(r"[{}]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, text, fontsize=14, color="#555555",
                 fontfamily="monospace")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    transparent=True, pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.error("Rendu texte brut échoué : %s", exc)
        return None


def formula_to_tk_image(latex: str, display: bool = True, max_height: int | None = None):
    """
    Retourne un objet PhotoImage Tkinter (ou None).
    Doit être conservé en référence pour éviter la GC.
    """
    if not latex or not latex.strip():
        return None
    try:
        from PIL import Image, ImageTk
        dpi = 180 if display else 210
        png_bytes = render_formula(latex, display, dpi)
        if png_bytes is None:
            return None
        img = Image.open(io.BytesIO(png_bytes))
        if max_height and img.height > max_height:
            ratio = max_height / img.height
            img = img.resize((max(1, int(img.width * ratio)), max_height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as exc:
        logger.error("Conversion PIL→Tkinter échouée : %s", exc)
        return None
