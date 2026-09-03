# services/onboarding.py — Le document de démonstration de la visite guidée.
#
# La visite guidée doit pouvoir montrer une lecture SANS que l'utilisateur ait
# encore quoi que ce soit dans sa bibliothèque. Elle emprunte donc un article
# embarqué dans l'application, le temps de la visite, puis le rend.
#
# « Le temps de la visite » est une contrainte, pas une facilité : le document
# apparaît dans la bibliothèque pendant qu'on l'explique, et il en a disparu
# quand la visite se termine. Personne ne doit se retrouver avec un faux
# document dans sa bibliothèque parce qu'il a fermé la fenêtre au mauvais
# moment — d'où `reconcile_demo_document()`, appelée à chaque démarrage.
#
# Ce module est le SEUL à connaître l'existence de ce document. En particulier
# il porte la seule suppression de document du dépôt, et elle ne prend pas d'id
# en paramètre : elle ne peut effacer que le document qu'elle a elle-même créé.
# On n'ouvre pas au passage un `DELETE /api/library/documents/{id}` générique,
# qui serait un pouvoir bien plus grand que ce que la visite demande.
#
# La session de lecture, elle, n'est PAS jouée ici : elle est entièrement
# scriptée côté frontend (`features/reader/demoScript.ts`), qui n'appelle ni
# `/api/session/*` ni le WebSocket du lecteur. Rien de la fausse session
# n'atteint donc le profil métacognitif — par construction, pas par un drapeau
# qu'il faudrait penser à respecter dans chaque écriture.
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from config.settings import ASSETS_DIR
from db import get_connection
from db.app_settings import delete_setting, get_setting, set_setting
from db.documents import get_document, update_document_digest, upsert_document
from pdf_viewer import page_renderer
from pdf_viewer.pdf_document import PdfDocument
from services import library

logger = logging.getLogger("services.onboarding")

__all__ = ["ensure_demo_document", "discard_demo_document", "reconcile_demo_document"]

#: Clé `app_settings` portant l'id du document emprunté. Sa présence signifie
#: « un document de démo existe et devra être rendu ».
_SETTING_KEY = "onboarding_doc_id"

# Ressources embarquées (lecture seule). Même motif que `server/config.py` pour
# `FRONTEND_DIST` : sous PyInstaller les données sont extraites sous _MEIPASS.
if getattr(sys, "frozen", False):
    RESOURCES_DIR = Path(getattr(sys, "_MEIPASS", ".")) / "resources"
else:
    RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"


def _bundled_pdf(lang: str) -> Path | None:
    """Le PDF de démo de la langue demandée, ou le FR à défaut, ou rien.

    « Ou rien » est un cas normal, pas une erreur : le dépôt peut être checkout
    sans ses ressources, et une visite doit se dérouler quand même — sans son
    chapitre lecture, mais sans planter (cf. `useTour.start`).
    """
    for candidate in (f"demo_{lang}.pdf", "demo_fr.pdf"):
        path = RESOURCES_DIR / candidate
        if path.is_file():
            return path
    logger.warning("Aucun PDF de démonstration dans %s", RESOURCES_DIR)
    return None


def _working_copy() -> Path:
    """Chemin stable du document emprunté, dans le dossier de données.

    On travaille sur une COPIE et non sur la ressource embarquée : sous
    PyInstaller, `_MEIPASS` est un dossier temporaire qui change à chaque
    lancement, alors que `documents.path` est persistant et sert de clé d'unicité
    (`upsert_document` fait un ON CONFLICT dessus). La copie rend aussi la
    suppression symétrique — on efface un fichier qu'on a écrit soi-même.
    """
    return Path(ASSETS_DIR) / "onboarding" / "demo.pdf"


def ensure_demo_document(lang: str = "fr", title: str = "", subject: str = "",
                         summary: str = "", keywords: list[str] | None = None) -> dict | None:
    """Prête le document de démonstration à la bibliothèque. Renvoie son détail.

    Renvoie `None` si aucune ressource n'est disponible : l'appelant saute alors
    le chapitre lecture de la visite plutôt que d'échouer.

    N'appelle volontairement PAS `orchestrator.import_pdf()`, qui est le chemin
    d'un vrai import, pour deux raisons :
      * il déclenche `generate_document_digest`, donc un appel LLM — inutile ici
        (la fiche est écrite à la main, juste en dessous) et qui laisserait la
        carte en `digest_status='pending'` pendant qu'on l'explique ;
      * la route `POST /api/library/import` refuserait de toute façon un chemin
        hors de `$HOME` (`server/security.py:import_path_allowed`).
    On recompose donc directement les briques qu'`import_pdf` assemble.
    """
    source = _bundled_pdf(lang)
    if source is None:
        return None

    target = _working_copy()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # `copyfile` et non `copy2` : on ne veut ni les permissions ni les dates
        # du fichier du bundle, seulement son contenu.
        shutil.copyfile(source, target)
    except OSError:
        logger.warning("Copie du document de démonstration impossible", exc_info=True)
        return None

    path = str(target.resolve())
    try:
        with PdfDocument(path) as pdf:
            page_count = pdf.page_count()
            has_toc = bool(pdf.toc())
    except Exception:
        logger.warning("Document de démonstration illisible : %s", path, exc_info=True)
        return None

    doc_id = upsert_document(path, title or target.stem, page_count, "pdfium_scroll", has_toc)
    # Fiche écrite à la main : la carte est complète tout de suite et hors ligne.
    update_document_digest(doc_id, subject=subject or None, summary=summary,
                           keywords=list(keywords or []))
    set_setting(_SETTING_KEY, str(doc_id))
    logger.info("Document de démonstration prêté id=%s (%s pages)", doc_id, page_count)
    return library.get_document(doc_id)


def demo_document_id() -> int | None:
    """L'id du document emprunté, s'il y en a un."""
    raw = get_setting(_SETTING_KEY)
    try:
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


def discard_demo_document() -> None:
    """Rend le document emprunté : ligne, caches et fichier. Idempotente.

    Une seule requête SQL suffit pour les données liées : `PRAGMA foreign_keys=ON`
    est posé par `db.get_connection()` et toutes les tables qui référencent
    `documents` sont en `ON DELETE CASCADE` (chapitres, pages en cache,
    questions, réponses, surlignages, sessions…). Seul `flashcards.document_id`
    est en `ON DELETE SET NULL`, ce qui est le bon comportement : une carte ne
    disparaît pas parce que son document a disparu.
    """
    doc_id = demo_document_id()
    if doc_id is None:
        return

    doc = get_document(doc_id)
    path = (doc or {}).get("path") or str(_working_copy())

    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    # Les PNG de pages rendues ne sont pas en base : ils vivent sous
    # `ASSETS_DIR/page_cache/<hash>` et personne ne les collecterait ensuite.
    try:
        page_renderer.clear_page_cache(path)
        page_renderer.page_cache_dir(path).rmdir()
    except OSError:
        pass  # cache absent ou non vide : sans conséquence.

    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.debug("Fichier de démonstration non supprimé : %s", path, exc_info=True)

    delete_setting(_SETTING_KEY)
    logger.info("Document de démonstration rendu id=%s", doc_id)


def reconcile_demo_document() -> None:
    """Garde-fou de démarrage : jette un document de démo qui aurait survécu.

    Une visite ne survit pas à un redémarrage — elle repart de sa première bulle.
    Un id encore posé au lancement est donc forcément le résidu d'une application
    fermée en plein milieu, et le document correspondant n'a rien à faire dans la
    bibliothèque de quelqu'un.
    """
    if demo_document_id() is None:
        return
    logger.info("Document de démonstration résiduel détecté au démarrage : nettoyage.")
    try:
        discard_demo_document()
    except Exception:  # pragma: no cover - un démarrage ne doit jamais échouer là-dessus
        logger.warning("Nettoyage du document de démonstration impossible", exc_info=True)
