# config/settings.py — Paramètres globaux MetaC-App
from pathlib import Path

# Racine du projet (V2/) : config/ → nwol/ → V2/
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:e2b"
OLLAMA_TIMEOUT = 60
OLLAMA_OPTIONS = {
    "num_ctx": 4096,
    "num_predict": 512,
    "temperature": 0.2,
}

OLLAMA_KEEP_ALIVE = "30m"

# Options spécifiques par type de tâche LLM
# Permet d'économiser du compute sur les tâches courtes sans sacrifier la précision des tâches complexes.
OLLAMA_TASK_OPTIONS: dict[str, dict] = {
    "curiosity_hook":           {"num_ctx": 2048, "num_predict": 180, "temperature": 0.1},
    "flashcard_tags":           {"num_ctx": 2048, "num_predict": 140, "temperature": 0.1},
    "session_summary":          {"num_ctx": 3072, "num_predict": 360, "temperature": 0.1},
    "question":                 {"num_ctx": 4096, "num_predict": 700, "temperature": 0.1},
    "evaluation":               {"num_ctx": 4096, "num_predict": 680, "temperature": 0.1},
    "rephrasing":               {"num_ctx": 4096, "num_predict": 560, "temperature": 0.1},
    "follow_up":                {"num_ctx": 4096, "num_predict": 560, "temperature": 0.1},
    "chapter_summary":          {"num_ctx": 4096, "num_predict": 520, "temperature": 0.1},
    "meta_cognition_questions": {"num_ctx": 4096, "num_predict": 300, "temperature": 0.1},
    "meta_cognition_analysis":  {"num_ctx": 4096, "num_predict": 320, "temperature": 0.1},
    "math_render":              {"num_ctx": 4096, "num_predict": 900, "temperature": 0.1},
    "schema_description":       {"num_ctx": 4096, "num_predict": 460, "temperature": 0.1},
    "table_description":        {"num_ctx": 4096, "num_predict": 520, "temperature": 0.1},
    "subject_detection":        {"num_ctx": 1024, "num_predict": 30,  "temperature": 0.1},
}

# Vitesse d'affichage caractère par caractère du texte généré par le LLM (ms/char)
LLM_CHAR_SPEED_MS = 15

# Lecture progressive
MIN_SPEED_MS = 10
MAX_SPEED_MS = 500
READING_SPEED_INITIAL_MS = MAX_SPEED_MS  # ms/caractère (vitesse la plus lente)
DEFAULT_SPEED_MS = READING_SPEED_INITIAL_MS
FIGURE_DISPLAY_PAUSE_MS = 3000

# Chapitres heuristiques (si pas de TOC)
DEFAULT_PAGES_PER_CHAPTER = 10

# Base de données
DB_PATH = str(_PROJECT_ROOT / "data" / "nwol.db")
DB_SCHEMA_VERSION = 15

# Logs
LOG_FILE = str(_PROJECT_ROOT / "logs" / "nwol.log")
LOG_MAX_BYTES = 1_000_000
LOG_BACKUP_COUNT = 5

# Assets temporaires générés depuis les PDFs (figures natives, etc.)
ASSETS_DIR = str(_PROJECT_ROOT / "nwol" / "assets")

# UI
UI_FONT_FAMILY = "Georgia"
UI_FONT_SIZE = 15
UI_MONO_FONT = "Courier"
UI_BG_COLOR = "#FFFFFF"
UI_FG_COLOR = "#1A1A1A"
UI_HEADING_COLOR = "#0D3B6E"
UI_LEFT_PANE_RATIO = 0.67

# Moteurs d'extraction (ordre cascade)
EXTRACTION_ENGINES = ["pymupdf_structured", "pymupdf"]

# Pipeline PDF ciblé par LLM
MAX_CONCURRENT_LLM_PDF_TASKS = 1
LLM_LAYOUT_TIMEOUT = 12
LLM_CROP_TIMEOUT = 18
LLM_LATEX_TIMEOUT = 15
LLM_CACHE_ENABLED = True
PDF_LLM_TIMEOUT_COOLDOWN = 75
PDF_LLM_MAX_ORDER_BLOCKS = 32
PDF_LLM_MAX_ORDER_ANCHORS = 10
PDF_READER_INITIAL_PAGES = 3
