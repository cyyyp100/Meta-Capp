# Tests de la page Brainstorming (REST + WebSocket avec LLM mocké).


# ── Fakes LLM (mêmes signatures callback que le code réel) ────────────────────

def _decide_no_search(history, user_message, on_success, on_error, model=None):
    on_success({"search": False, "queries": []})


def _decide_with_search(history, user_message, on_success, on_error, model=None):
    on_success({"search": True, "queries": ["vélo"]})


def _answer_fake(context, on_success, on_error, model=None):
    sources = context.get("sources") or []
    on_success(f"Réponse de test ({len(sources)} source(s)) : {context.get('user_message')}")


def _summarize_noop(previous_summary, new_messages, on_success, on_error, model=None):
    on_success("résumé de test")


# ── REST ──────────────────────────────────────────────────────────────────────

def test_brainstorm_crud(client):
    created = client.post("/api/brainstorming", json={"title": "Mon sujet"}).json()
    assert created["id"] >= 1
    assert created["title"] == "Mon sujet"

    listing = client.get("/api/brainstorming/discussions").json()
    assert any(d["id"] == created["id"] for d in listing)

    detail = client.get(f"/api/brainstorming/{created['id']}/messages").json()
    assert detail["title"] == "Mon sujet"
    assert detail["messages"] == []

    client.delete(f"/api/brainstorming/{created['id']}")
    assert client.get("/api/brainstorming/discussions").json() == []


def test_brainstorm_messages_404_on_unknown(client):
    assert client.get("/api/brainstorming/999/messages").status_code == 404


# ── WebSocket ─────────────────────────────────────────────────────────────────

def test_brainstorm_ws_answer_and_persist(client, monkeypatch):
    import services.brainstorm as svc

    monkeypatch.setattr(svc, "decide_brainstorm_search_async", _decide_no_search)
    monkeypatch.setattr(svc, "answer_brainstorm_async", _answer_fake)
    monkeypatch.setattr(svc, "summarize_brainstorm_async", _summarize_noop)

    did = client.post("/api/brainstorming", json={}).json()["id"]

    with client.websocket_connect(f"/api/brainstorming/{did}/stream") as ws:
        ws.send_json({"type": "ask", "question": "Idée sur le vélo"})
        assert ws.receive_json()["type"] == "loading"
        answer = ws.receive_json()
        assert answer["type"] == "answer"
        assert "vélo" in answer["answer"]
        assert answer["sources"] == []

    detail = client.get(f"/api/brainstorming/{did}/messages").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    # Auto-titre depuis la 1re question (discussion créée sans titre).
    assert detail["title"].startswith("Idée sur le vélo")


def test_brainstorm_ws_runs_db_search(client, monkeypatch):
    import services.brainstorm as svc

    captured = {}

    def _spy_search(query, *a, **k):
        captured["query"] = query
        return [{"source_type": "highlight", "doc_title": "PDF", "page": 3, "snippet": "passage vélo"}]

    monkeypatch.setattr(svc, "decide_brainstorm_search_async", _decide_with_search)
    monkeypatch.setattr(svc, "answer_brainstorm_async", _answer_fake)
    monkeypatch.setattr(svc, "summarize_brainstorm_async", _summarize_noop)
    monkeypatch.setattr(svc.brainstorm_search, "search_user_db", _spy_search)

    did = client.post("/api/brainstorming", json={"title": "Vélo"}).json()["id"]

    with client.websocket_connect(f"/api/brainstorming/{did}/stream") as ws:
        ws.send_json({"type": "ask", "question": "Parle-moi de vélo"})
        events = []
        # loading -> scanning(on) -> scanning(off) -> answer (ordre exact non garanti
        # pour les scanning, mais answer arrive en dernier).
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] == "answer":
                break

    types = [e["type"] for e in events]
    assert "loading" in types
    assert "scanning" in types
    assert captured["query"] == "vélo"
    answer = events[-1]
    assert answer["sources"] and answer["sources"][0]["source_type"] == "highlight"
    assert "1 source" in answer["answer"]


def test_brainstorm_question_is_bounded(client, monkeypatch):
    """S4 : une question démesurée est TRONQUÉE, pas refusée.

    Le WebSocket du lecteur borne ses entrées depuis toujours (`ReaderMessage`) ;
    celui du brainstorming ne le faisait pas, et la chaîne partait telle quelle
    dans un prompt. On tronque plutôt que de fermer le socket : une question trop
    longue reste une question, et fermer ferait perdre la discussion."""
    from server.routers import brainstorming as router
    from services import brainstorm as svc

    seen: dict = {}

    def _capture(discussion_id, question, on_answer, on_error, on_scanning):
        seen["question"] = question
        on_answer({"answer": "ok", "sources": []})

    monkeypatch.setattr(svc, "handle_message", _capture)

    did = client.post("/api/brainstorming", json={"title": "Bornage"}).json()["id"]
    with client.websocket_connect(f"/api/brainstorming/{did}/stream") as ws:
        ws.send_json({"type": "ask", "question": "a" * 10_000})
        while ws.receive_json()["type"] != "answer":
            pass

    assert len(seen["question"]) == router._MAX_QUESTION_CHARS


def test_brainstorm_ignores_non_dict_message(client, monkeypatch):
    """Un message JSON qui n'est pas un objet ne doit pas faire tomber le socket.

    `msg.get(...)` sur une liste lève un AttributeError qui remontait jusqu'au
    gestionnaire d'exception du WebSocket et fermait le canal."""
    from services import brainstorm as svc

    def _answer(discussion_id, question, on_answer, on_error, on_scanning):
        on_answer({"answer": "ok", "sources": []})

    monkeypatch.setattr(svc, "handle_message", _answer)

    did = client.post("/api/brainstorming", json={"title": "Robuste"}).json()["id"]
    with client.websocket_connect(f"/api/brainstorming/{did}/stream") as ws:
        ws.send_json(["pas", "un", "objet"])   # ignoré
        ws.send_json({"type": "inconnu"})      # ignoré
        # Le socket est toujours vivant : un `ask` valide obtient sa réponse.
        ws.send_json({"type": "ask", "question": "toujours là ?"})
        while ws.receive_json()["type"] != "answer":
            pass


# ── Recherche en base : pertinence, diversité, rotation ──────────────────────
#
# Avant, chaque source rendait « les 3 premiers matchs dans l'ordre des id », le
# tout concaténé dans un ordre fixe : la réponse citait toujours les 3 surlignages
# et les 3 flashcards les plus récents, une Q&R ou un document n'apparaissaient
# jamais, et reposer la même question rendait la MÊME liste au caractère près.

def _seed_highlights(n: int, word: str = "photosynthese") -> int:
    from db.documents import upsert_document
    from db.reader_highlights import add_highlight

    doc_id = upsert_document(
        path="/tmp/bio.pdf", filename="bio.pdf", page_count=n + 1,
        engine="test", has_toc=False, subject="biologie",
    )
    for i in range(n):
        add_highlight(doc_id, page=i + 1, quote=f"Extrait {i} sur la {word}.", rects=[])
    return doc_id


def test_search_is_not_limited_to_the_newest_rows(client):
    """Un surlignage ancien doit pouvoir être cité s'il répond à la question."""
    from services.brainstorm_search import search_user_db

    _seed_highlights(80)
    seen: set[str] = set()
    for _ in range(40):
        seen.update(item["snippet"] for item in search_user_db("photosynthèse"))
    # Avant : exactement 3 extraits, toujours les mêmes (les 3 plus récents).
    assert len(seen) > 10


def test_search_spreads_across_source_types(client):
    """Plafond par type : une Q&R et un document ne sont plus affamés par les surlignages."""
    from db.flashcards import save_flashcard
    from db.questions import save_assistant_exchange
    from db.user import DEFAULT_USER_ID
    from services.brainstorm_search import search_user_db

    doc_id = _seed_highlights(30)
    for i in range(10):
        save_flashcard(DEFAULT_USER_ID, None, f"photosynthese {i} ?", "réponse", document_id=doc_id)
        save_assistant_exchange(doc_id, i + 1, f"photosynthese, comment ? {i}", "explication")

    types_seen: set[str] = set()
    for _ in range(20):
        types_seen.update(item["source_type"] for item in search_user_db("photosynthèse"))
    assert {"highlight", "flashcard", "qa"} <= types_seen

    for _ in range(20):
        counts: dict[str, int] = {}
        for item in search_user_db("photosynthèse"):
            counts[item["source_type"]] = counts.get(item["source_type"], 0) + 1
        assert max(counts.values()) <= 2, counts


def test_search_favours_the_most_relevant_without_locking_onto_it(client):
    """Deux termes trouvés pèsent bien plus qu'un seul, sans rendre le tirage figé.

    L'extrait qui touche TOUTE la requête est en concurrence avec 30 extraits qui
    n'en touchent que la moitié, pour 2 places. Un tirage neutre le sortirait dans
    ~6 % des tours (2/31) ; la pente de pertinence doit le hisser bien au-dessus.

    Mais l'assertion s'arrête là : exiger qu'il sorte à TOUS les tours reviendrait
    à réclamer le déterminisme que cette sélection existe justement pour casser.
    """
    from db.documents import upsert_document
    from db.reader_highlights import add_highlight
    from services.brainstorm_search import search_user_db

    doc_id = upsert_document(
        path="/tmp/bio2.pdf", filename="bio2.pdf", page_count=50,
        engine="test", has_toc=False, subject="biologie",
    )
    add_highlight(doc_id, page=1, quote="La photosynthese et la chlorophylle ensemble.", rects=[])
    for i in range(30):
        add_highlight(doc_id, page=i + 2, quote=f"Juste la photosynthese, note {i}.", rects=[])

    hits = 0
    for _ in range(100):
        snippets = [item["snippet"] for item in search_user_db("photosynthèse chlorophylle")]
        hits += any("chlorophylle" in s for s in snippets)
    assert hits > 20, f"le plus pertinent ne sort que {hits} fois sur 100 (hasard : ~6)"


def test_already_cited_sources_are_damped(client):
    """Une source déjà citée recule sans être exclue."""
    from services.brainstorm_search import search_user_db, source_key

    _seed_highlights(20)
    first = search_user_db("photosynthèse")
    assert first
    damp = {source_key(item) for item in first}

    repeats = 0
    for _ in range(30):
        again = search_user_db("photosynthèse", damp_keys=damp)
        repeats += sum(source_key(item) in damp for item in again)
    # Sans amortissement, les mêmes extraits reviendraient massivement.
    assert repeats < 30, f"{repeats} re-citations sur 30 tours"


def test_search_returns_nothing_without_usable_terms(client):
    from services.brainstorm_search import search_user_db

    _seed_highlights(5)
    assert search_user_db("le la les") == []
    assert search_user_db("") == []
    assert search_user_db("photosynthèse", limit=0) == []
