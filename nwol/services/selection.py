# services/selection.py — « Beaucoup de lignes correspondent : en choisir quelques-unes, bien. »
#
# Maison unique de la politique de sélection. Avant, chaque appelant réinventait
# la sienne — `random.choices` dans db/flashcards, `random.shuffle` deux fois dans
# services/quiz, « prendre les 3 premiers dans l'ordre des id » dans
# services/brainstorm_search — et toutes convergeaient vers le même défaut : sur une
# base bien remplie, seules les lignes LES PLUS RÉCENTES sortaient, à l'identique
# d'une session à l'autre.
#
# Trois primitives, et rien d'autre :
#   - `decay` / `cooldown` : le temps qui passe, en poids multiplicatif ;
#   - `weighted_sample`    : tirer k éléments SANS remise, proportionnellement ;
#   - `diversified_take`   : ne pas laisser une seule catégorie rafler la place.
#
# Le principe partout : AMORTIR, jamais EXCLURE. Une ligne servie à l'instant voit
# son poids s'effondrer, mais reste tirable — sinon une notion ratée ne pourrait
# plus revenir vite, ce qui est exactement le contraire de la répétition espacée.
from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

__all__ = [
    "decay",
    "cooldown",
    "age_days",
    "weighted_sample",
    "diversified_take",
    "relevance",
]

# En dessous, un poids est traité comme nul (jamais tiré) : évite qu'un flottant
# dénormalisé ne produise une clé de tri absurde dans `weighted_sample`.
_MIN_WEIGHT = 1e-12


def age_days(timestamp, now: datetime | None = None) -> float | None:
    """Âge en jours d'un horodatage SQLite (`'...T...'` ou `'... ...'`), ou None.

    Les deux formats coexistent en base : `update_review` écrit `last_reviewed`
    en ISO (avec « T ») tandis que `due_at` passe par `_sql_datetime` (espace).
    `datetime.fromisoformat` accepte les deux, mais pas une valeur vide ni None.
    """
    if not timestamp:
        return None
    if isinstance(timestamp, datetime):
        parsed = timestamp
    else:
        try:
            parsed = datetime.fromisoformat(str(timestamp))
        except (TypeError, ValueError):
            return None
    reference = now or datetime.now()
    if parsed.tzinfo is not None and reference.tzinfo is None:
        parsed = parsed.replace(tzinfo=None)
    return max(0.0, (reference - parsed).total_seconds() / 86400.0)


def decay(days: float | None, half_life_days: float) -> float:
    """Décroissance exponentielle : 1.0 à l'instant t, 0.5 après une demi-vie.

    Résultat dans ``]0, 1]``. Un âge inconnu (``None``) est traité comme très
    ancien plutôt que comme neuf : une ligne sans horodatage exploitable ne doit
    pas hériter de la priorité du matériel frais.
    """
    if half_life_days <= 0:
        return 1.0
    if days is None:
        return 0.0
    return math.exp(-max(0.0, days) * math.log(2) / half_life_days)


def cooldown(last_seen, half_life_days: float, floor: float, now: datetime | None = None) -> float:
    """Amortissement d'une ligne récemment servie. Résultat dans ``[floor, 1.0]``.

    Vaut ``floor`` juste après l'exposition et remonte vers 1.0 en une demi-vie.
    Une ligne jamais servie (``last_seen`` vide) vaut 1.0 — c'est ce qui fait
    remonter le stock ancien qu'aucune session n'a encore touché.
    """
    days = age_days(last_seen, now=now)
    if days is None:  # jamais vue : aucun amortissement
        return 1.0
    floor = min(max(floor, 0.0), 1.0)
    return floor + (1.0 - floor) * (1.0 - decay(days, half_life_days))


def relevance(folded_text: str, folded_terms: Sequence[str]) -> tuple[int, int]:
    """Pertinence d'un texte pour des termes déjà repliés → ``(distincts, total)``.

    Convention historique de `services.pdf_rag.rank_chunks` : nombre de termes
    DISTINCTS présents d'abord (un texte qui touche trois notions bat celui qui
    répète la même dix fois), occurrences totales pour départager. Elle était
    réécrite à l'identique dans `services.quiz` ; elle vit ici désormais, et les
    deux appelants la partagent.
    """
    if not folded_terms or not folded_text:
        return (0, 0)
    distinct = 0
    total = 0
    for term in folded_terms:
        count = folded_text.count(term)
        if count:
            distinct += 1
            total += count
    return (distinct, total)


def weighted_sample(
    items: Sequence[T],
    weights: Sequence[float],
    k: int,
    rng: random.Random | None = None,
) -> list[T]:
    """``k`` éléments tirés proportionnellement aux poids, SANS remise.

    Algorithme d'Efraimidis–Spirakis : chaque élément reçoit la clé
    ``U ** (1 / w)`` et on garde les ``k`` plus grandes. Un seul passage, pas de
    doublon possible, et la probabilité de figurer dans le tirage suit bien les
    poids.

    C'est ce qui remplace le `random.choices` (tirage AVEC remise) qui régnait
    dans le sas d'entrée : là-bas, chaque collision était réparée en rebouchant
    dans l'ordre de récence, ce qui ramenait sournoisement le tirage vers « les
    cartes les plus récentes » — précisément ce qu'il devait éviter.

    Les poids nuls ou négatifs ne sont jamais tirés. Si moins de ``k`` éléments
    ont un poids exploitable, on renvoie ce qui existe.
    """
    if k <= 0 or not items:
        return []
    draw = rng or random
    keyed: list[tuple[float, int, T]] = []
    for index, item in enumerate(items):
        try:
            weight = float(weights[index])
        except (IndexError, TypeError, ValueError):
            continue
        if not math.isfinite(weight) or weight <= _MIN_WEIGHT:
            continue
        # log(U) / w est équivalent à U ** (1/w) pour le tri, sans risque
        # d'underflow quand w est petit (U ** 1e6 vaut 0.0 en flottant).
        u = draw.random() or _MIN_WEIGHT
        keyed.append((math.log(u) / weight, index, item))
    keyed.sort(key=lambda row: row[0], reverse=True)
    return [item for (_key, _index, item) in keyed[:k]]


def diversified_take(
    items: Iterable[T],
    key_fn: Callable[[T], object],
    per_key_cap: int,
    total: int,
) -> list[T]:
    """``total`` éléments servis à tour de rôle par catégorie, ``per_key_cap`` max chacune.

    L'ordre reçu fait foi À L'INTÉRIEUR d'une catégorie (l'appelant l'a déjà
    classé ou tiré) ; c'est la répartition ENTRE catégories qui est corrigée ici.
    Sans ce tour de rôle, une catégorie bien fournie rafle la place par la seule
    loi des grands nombres — c'est ainsi que le brainstorming ne citait jamais
    ni une ancienne Q&R ni un document, toujours mangés par les surlignages.

    Si le quota reste inemployé (moins de catégories que de places), un second
    passage remplit avec le reste, plafond par catégorie compris.
    """
    if total <= 0:
        return []
    buckets: dict[object, list[T]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)

    picked: list[T] = []
    taken: dict[object, int] = {key: 0 for key in buckets}
    # Premier passage : tour de rôle sous le plafond. Second passage : le plafond
    # est levé pour ne pas rendre une liste plus courte que demandé alors qu'il
    # reste des candidats.
    for cap in (per_key_cap, total):
        if cap <= 0:
            continue
        drained = False
        while len(picked) < total and not drained:
            drained = True
            for key, bucket in buckets.items():
                if len(picked) >= total or taken[key] >= cap or not bucket:
                    continue
                picked.append(bucket.pop(0))
                taken[key] += 1
                drained = False
        if len(picked) >= total:
            break
    return picked
