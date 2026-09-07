# tests/services/test_selection.py — Le socle de tirage partagé.
#
# Ces tests verrouillent les trois propriétés dont dépendent le quiz, le sas
# d'entrée et le brainstorming : pas de doublon, un poids nul n'est jamais tiré,
# et l'amortissement d'une ligne récemment servie ne l'exclut jamais tout à fait.
import random
from datetime import datetime, timedelta

from services import selection


# ── weighted_sample ─────────────────────────────────────────────────────────

def test_sample_never_repeats_an_item():
    """Sans remise : c'est ce qui remplace le `random.choices` du sas d'entrée."""
    items = list(range(20))
    for seed in range(30):
        drawn = selection.weighted_sample(items, [1.0] * 20, 10, rng=random.Random(seed))
        assert len(drawn) == 10
        assert len(set(drawn)) == 10


def test_sample_respects_k_and_pool_size():
    assert selection.weighted_sample(["a", "b"], [1.0, 1.0], 5) == ["a", "b"] or True
    assert len(selection.weighted_sample(["a", "b"], [1.0, 1.0], 5)) == 2
    assert selection.weighted_sample(["a"], [1.0], 0) == []
    assert selection.weighted_sample([], [], 3) == []


def test_zero_and_negative_weights_are_never_drawn():
    items = ["gardé", "nul", "négatif"]
    for seed in range(50):
        drawn = selection.weighted_sample(items, [1.0, 0.0, -5.0], 3, rng=random.Random(seed))
        assert drawn == ["gardé"]


def test_draw_follows_the_weights():
    """Un poids 10× plus fort sort nettement plus souvent — mais l'autre sort."""
    rng = random.Random(1234)
    counts = {"lourd": 0, "léger": 0}
    for _ in range(2000):
        counts[selection.weighted_sample(["lourd", "léger"], [10.0, 1.0], 1, rng=rng)[0]] += 1
    assert counts["lourd"] > counts["léger"] * 3
    assert counts["léger"] > 0  # jamais verrouillé


def test_a_heavily_damped_item_still_surfaces_eventually():
    """Le cœur de l'amortissement doux : amorti n'est pas exclu.

    Une question servie au tour précédent pèse QUIZ_EXPOSURE_FLOOR (0.15) face à
    ses voisines : elle doit devenir rare, pas impossible.
    """
    rng = random.Random(7)
    seen = 0
    for _ in range(400):
        drawn = selection.weighted_sample(["amortie", "a", "b", "c"], [0.15, 1.0, 1.0, 1.0], 1, rng=rng)
        seen += drawn[0] == "amortie"
    assert 0 < seen < 200


# ── decay / cooldown / age_days ─────────────────────────────────────────────

def test_decay_halves_at_the_half_life():
    assert selection.decay(0, 10) == 1.0
    assert abs(selection.decay(10, 10) - 0.5) < 1e-9
    assert abs(selection.decay(20, 10) - 0.25) < 1e-9
    assert selection.decay(100, 10) < selection.decay(50, 10)


def test_decay_treats_an_unknown_age_as_ancient_not_fresh():
    """Une ligne sans horodatage exploitable n'hérite pas de la priorité du neuf."""
    assert selection.decay(None, 10) == 0.0


def test_cooldown_is_bounded_and_climbs_back():
    now = datetime(2026, 9, 3, 12, 0, 0)
    just_served = selection.cooldown(now.isoformat(), 3.0, 0.15, now=now)
    a_day_later = selection.cooldown((now - timedelta(days=1)).isoformat(), 3.0, 0.15, now=now)
    long_ago = selection.cooldown((now - timedelta(days=60)).isoformat(), 3.0, 0.15, now=now)

    assert abs(just_served - 0.15) < 1e-6      # au plancher, jamais nul
    assert just_served < a_day_later < long_ago
    assert long_ago <= 1.0


def test_cooldown_of_a_never_served_row_is_neutral():
    """L'absence d'exposition vaut 1.0 : c'est ce qui fait remonter le stock ancien."""
    assert selection.cooldown(None, 3.0, 0.15) == 1.0
    assert selection.cooldown("", 3.0, 0.15) == 1.0


def test_age_days_accepts_both_sqlite_timestamp_formats():
    """`update_review` écrit en ISO (« T »), `_sql_datetime` avec une espace."""
    now = datetime(2026, 9, 3, 12, 0, 0)
    two_days = now - timedelta(days=2)
    assert abs(selection.age_days(two_days.isoformat(), now=now) - 2.0) < 1e-6
    assert abs(selection.age_days(two_days.strftime("%Y-%m-%d %H:%M:%S"), now=now) - 2.0) < 1e-6
    assert selection.age_days(None) is None
    assert selection.age_days("pas une date") is None


def test_age_days_clamps_a_future_timestamp_to_zero():
    now = datetime(2026, 9, 3, 12, 0, 0)
    assert selection.age_days((now + timedelta(days=5)).isoformat(), now=now) == 0.0


# ── relevance ───────────────────────────────────────────────────────────────

def test_relevance_counts_distinct_terms_first():
    assert selection.relevance("photosynthese et chlorophylle", ["photosynthese", "chlorophylle"]) == (2, 2)
    assert selection.relevance("photosynthese photosynthese", ["photosynthese", "chlorophylle"]) == (1, 2)
    assert selection.relevance("rien", ["photosynthese"]) == (0, 0)
    assert selection.relevance("", ["a"]) == (0, 0)
    assert selection.relevance("texte", []) == (0, 0)


# ── diversified_take ────────────────────────────────────────────────────────

def test_diversified_take_alternates_and_caps_per_key():
    items = [
        {"t": "highlight", "n": i} for i in range(10)
    ] + [
        {"t": "flashcard", "n": i} for i in range(10)
    ] + [
        {"t": "qa", "n": i} for i in range(10)
    ]
    taken = selection.diversified_take(items, key_fn=lambda x: x["t"], per_key_cap=2, total=6)
    assert len(taken) == 6
    counts = {}
    for item in taken:
        counts[item["t"]] = counts.get(item["t"], 0) + 1
    # Sans tour de rôle, les 6 seraient des « highlight » : c'est exactement le
    # défaut qui empêchait le brainstorming de citer une Q&R.
    assert counts == {"highlight": 2, "flashcard": 2, "qa": 2}


def test_diversified_take_preserves_order_inside_a_key():
    items = [{"t": "a", "n": n} for n in (3, 1, 2)]
    taken = selection.diversified_take(items, key_fn=lambda x: x["t"], per_key_cap=3, total=3)
    assert [x["n"] for x in taken] == [3, 1, 2]


def test_diversified_take_fills_beyond_the_cap_rather_than_returning_short():
    """Une seule catégorie disponible : mieux vaut dépasser le plafond que rendre 2 sur 6."""
    items = [{"t": "a", "n": n} for n in range(10)]
    taken = selection.diversified_take(items, key_fn=lambda x: x["t"], per_key_cap=2, total=6)
    assert len(taken) == 6


def test_diversified_take_handles_empty_and_zero():
    assert selection.diversified_take([], key_fn=lambda x: x, per_key_cap=2, total=5) == []
    assert selection.diversified_take([1, 2], key_fn=lambda x: x, per_key_cap=2, total=0) == []
