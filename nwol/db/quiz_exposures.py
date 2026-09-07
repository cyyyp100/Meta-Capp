# db/quiz_exposures.py — Mémoire des questions déjà servies en quiz (schéma v28).
#
# Deux opérations, et rien d'autre : lire l'exposition d'un lot de questions,
# enregistrer celle d'une session qui part. La POLITIQUE (à quel point amortir
# une question récemment posée) vit dans `services/quiz.py` et `services/selection.py` :
# ici on ne fait que stocker « quand » et « combien de fois ».
from __future__ import annotations

import logging

from db import get_connection
from db.user import DEFAULT_USER_ID, ensure_default_user

logger = logging.getLogger("DB.quiz_exposures")

# SQLite plafonne le nombre de paramètres liés d'une requête (999 sur les vieilles
# compilations). Le vivier du quiz vaut QUIZ_SEARCH_POOL=400, mais on découpe
# quand même : c'est le genre de limite qui se rappelle au mauvais moment.
_CHUNK = 400


def get_exposures(user_id: int, question_ids) -> dict[int, dict]:
    """``{question_id: {"times_served", "last_served_at"}}`` pour les ids demandés.

    Une question absente du résultat n'a jamais été servie — l'appelant traite ce
    cas comme « aucun amortissement », pas comme une erreur.
    """
    ids = [int(qid) for qid in question_ids if qid is not None]
    if not ids:
        return {}
    conn = get_connection()
    out: dict[int, dict] = {}
    for start in range(0, len(ids), _CHUNK):
        chunk = ids[start : start + _CHUNK]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""SELECT question_id, times_served, last_served_at
                FROM quiz_exposures
                WHERE user_id=? AND question_id IN ({placeholders})""",
            (user_id or DEFAULT_USER_ID, *chunk),
        ).fetchall()
        for row in rows:
            out[int(row["question_id"])] = {
                "times_served": int(row["times_served"] or 0),
                "last_served_at": row["last_served_at"],
            }
    return out


def record_exposures(user_id: int, items) -> None:
    """Marque des questions comme servies maintenant. ``items`` : ``[(id, source)]``.

    Appelé au moment où la session part vers le client, pas à la correction : une
    session abandonnée après trois questions doit quand même faire tourner le
    stock au tour suivant.
    """
    rows = [
        (user_id or DEFAULT_USER_ID, int(qid), str(source or "reading"))
        for qid, source in items
        if qid is not None
    ]
    if not rows:
        return
    ensure_default_user()
    conn = get_connection()
    with conn:
        # `localtime` et non l'UTC de `datetime('now')` : l'âge est calculé côté
        # Python contre `datetime.now()` (heure locale), comme le fait déjà
        # `db/flashcards.py` pour `due_at` / `last_reviewed`. Mélanger les deux
        # ferait paraître « vieille de deux heures » une question servie à l'instant.
        conn.executemany(
            """INSERT INTO quiz_exposures (user_id, question_id, source, times_served, last_served_at)
               VALUES (?, ?, ?, 1, datetime('now', 'localtime'))
               ON CONFLICT(user_id, question_id) DO UPDATE SET
                   times_served   = times_served + 1,
                   last_served_at = datetime('now', 'localtime'),
                   source         = excluded.source""",
            rows,
        )
    logger.debug("Exposition quiz enregistrée pour %d questions", len(rows))
