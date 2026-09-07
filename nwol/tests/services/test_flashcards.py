import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    import db
    from db import close_connection

    close_connection()
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "nwol.db"))
    from db.schema import initialize_schema

    initialize_schema()
    yield
    close_connection()


def test_create_list_and_filter(fresh_db):
    from services.flashcards import create_flashcard, list_flashcards

    cid = create_flashcard(front="2+2 ?", back="4", tags=["maths"], difficulty=1)
    assert isinstance(cid, int) and cid > 0

    cards = list_flashcards()
    assert len(cards) == 1
    card = cards[0]
    assert card["front"] == "2+2 ?"
    assert card["back"] == "4"
    assert "maths" in card["tags"]
    assert card["difficulty"] == 1
    assert card["source"] == "manual"

    # Le filtre par difficulté restreint la liste.
    assert len(list_flashcards(difficulty=1)) == 1
    assert len(list_flashcards(difficulty=3)) == 0


def test_existing_tags_aggregates(fresh_db):
    from services.flashcards import create_flashcard, existing_tags

    create_flashcard(front="a", back="b", tags=["algèbre", "maths"])
    create_flashcard(front="c", back="d", tags=["maths", "géométrie"])
    tags = existing_tags()
    assert set(tags) >= {"algèbre", "maths", "géométrie"}


def test_review_updates_due_date(fresh_db):
    from db.flashcards import get_flashcard
    from services.flashcards import create_flashcard, review_flashcard

    cid = create_flashcard(front="q", back="r")
    before = get_flashcard(cid)
    review_flashcard(cid, "correct")
    after = get_flashcard(cid)
    assert after["review_count"] == before["review_count"] + 1
    assert after["last_verdict"] == "correct"
    # Verdict "correct" => intervalle ×2.5 (s'allonge).
    assert float(after["interval_days"]) > float(before["interval_days"])


def test_delete_batch(fresh_db):
    from services.flashcards import create_flashcard, delete_flashcards, list_flashcards

    ids = [create_flashcard(front=f"q{i}", back=f"r{i}") for i in range(3)]
    removed = delete_flashcards(ids[:2])
    assert removed == 2
    remaining = list_flashcards()
    assert len(remaining) == 1
    assert remaining[0]["id"] == ids[2]


def test_fallback_tags_no_llm(fresh_db):
    from services.flashcards import fallback_tags

    tags = fallback_tags("Théorème de Pythagore", "a² + b² = c²", existing_tags=[])
    assert isinstance(tags, list)
    assert all(isinstance(tag, str) for tag in tags)


# ── Sas d'entrée : le tirage d'échauffement ─────────────────────────────────
#
# Trois défauts d'origine, tous ramenant l'échauffement aux cartes les plus
# récentes : un `LIMIT 60` sur `created_at DESC`, une demi-vie de récence de 7
# jours, et un `random.choices` AVEC remise dont les collisions étaient rebouchées
# dans l'ordre de récence.

def _seed_cards(n: int) -> list[int]:
    from services.flashcards import create_flashcard

    return [create_flashcard(front=f"q{i}", back=f"r{i}") for i in range(n)]


def test_session_start_reaches_beyond_the_sixty_newest(fresh_db):
    """Une carte hors des 60 plus récentes doit pouvoir sortir en échauffement."""
    from db.flashcards import get_session_start_cards

    ids = _seed_cards(100)
    oldest_forty = set(ids[:40])  # créées en premier => hors des 60 plus récentes

    seen: set[int] = set()
    for _ in range(60):
        seen.update(card["id"] for card in get_session_start_cards(n=5))
    # Avant : l'intersection était vide, quel que soit le nombre de tirages.
    assert seen & oldest_forty


def test_session_start_never_returns_the_same_card_twice(fresh_db):
    """Tirage sans remise : plus de doublon à réparer, donc plus de rebouchage biaisé."""
    from db.flashcards import get_session_start_cards

    _seed_cards(8)
    for _ in range(40):
        cards = get_session_start_cards(n=5)
        ids = [card["id"] for card in cards]
        assert len(ids) == len(set(ids))
        assert len(ids) == 5


def test_session_start_damps_cards_just_reviewed(fresh_db):
    """`last_reviewed` est écrit par l'échauffement lui-même : il doit être lu."""
    from db.flashcards import get_session_start_cards
    from services.flashcards import review_flashcard

    ids = _seed_cards(20)
    just_seen = set(ids[:5])
    for cid in just_seen:
        review_flashcard(cid, "partial")  # ce que fait WarmUp sur chaque carte montrée

    hits = 0
    for _ in range(60):
        hits += sum(card["id"] in just_seen for card in get_session_start_cards(n=5))
    # Tirage neutre : 5 cartes sur 20 pour 5 places => ~75 sur 60 tours.
    # Amorties à FLASHCARD_REVIEW_FLOOR, elles doivent nettement reculer,
    # sans disparaître (amortir, pas exclure).
    assert 0 < hits < 45, f"cartes déjà vues servies {hits} fois"


def test_session_start_returns_everything_when_the_stock_is_small(fresh_db):
    from db.flashcards import get_session_start_cards

    _seed_cards(3)
    assert len(get_session_start_cards(n=5)) == 3


def test_session_start_handles_an_empty_library(fresh_db):
    from db.flashcards import get_session_start_cards

    assert get_session_start_cards(n=5) == []


def test_due_cards_keep_absolute_priority(fresh_db):
    """La répétition espacée passe avant le tirage pondéré."""
    from datetime import datetime, timedelta

    from db import get_connection
    from db.flashcards import get_session_start_cards

    ids = _seed_cards(30)
    overdue = ids[0]
    with get_connection() as conn:
        conn.execute(
            "UPDATE flashcards SET due_at=? WHERE id=?",
            ((datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"), overdue),
        )
    for _ in range(10):
        cards = get_session_start_cards(n=5)
        assert cards[0]["id"] == overdue
