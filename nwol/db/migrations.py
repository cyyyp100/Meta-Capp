# db/migrations.py — Migrations SQLite idempotentes
from __future__ import annotations

import logging

from config.settings import DB_SCHEMA_VERSION

logger = logging.getLogger("DB.migrations")


TARGET_SCHEMA_VERSION = DB_SCHEMA_VERSION


def run_migrations(conn) -> None:
    """Applique les migrations nécessaires jusqu'à la version cible."""
    current = _current_version(conn)
    if current < 2 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v2(conn)
        _set_version(conn, 2)
        current = 2

    if current < 3 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v3(conn)
        _set_version(conn, 3)
        current = 3

    if current < 4 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v4(conn)
        _set_version(conn, 4)
        current = 4

    if current < 5 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v5(conn)
        _set_version(conn, 5)
        current = 5

    if current < 6 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v6(conn)
        _set_version(conn, 6)
        current = 6

    if current < 7 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v7(conn)
        _set_version(conn, 7)
        current = 7

    if current < 8 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v8(conn)
        _set_version(conn, 8)
        current = 8

    if current < 9 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v9(conn)
        _set_version(conn, 9)
        current = 9

    if current < 10 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v10(conn)
        _set_version(conn, 10)
        current = 10

    if current < 11 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v11(conn)
        _set_version(conn, 11)
        current = 11

    if current < 12 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v12(conn)
        _set_version(conn, 12)
        current = 12

    if current < 13 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v13(conn)
        _set_version(conn, 13)
        current = 13

    if current < 14 <= TARGET_SCHEMA_VERSION:
        _migrate_to_v14(conn)
        _set_version(conn, 14)
        current = 14

    if current < TARGET_SCHEMA_VERSION:
        _set_version(conn, TARGET_SCHEMA_VERSION)


def _current_version(conn) -> int:
    row = conn.execute(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return int(row["version"]) if row else 1


def _set_version(conn, version: int) -> None:
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    logger.info("Version schéma SQLite : v%s", version)


def _migrate_to_v2(conn) -> None:
    logger.info("Migration SQLite v2 démarrée")

    _ensure_column(conn, "documents", "doc_type", "TEXT DEFAULT 'book'")
    _ensure_column(conn, "questions", "llm_model", "TEXT")
    _ensure_column(conn, "reading_sessions", "user_id", "INTEGER DEFAULT 1")
    _ensure_column(conn, "reading_sessions", "duration_s", "INTEGER")
    _ensure_column(conn, "reading_sessions", "chapters_completed", "TEXT")

    _ensure_v2_tables(conn)
    logger.info("Migration SQLite v2 terminée")


def _ensure_column(conn, table: str, column: str, definition: str) -> None:
    columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        logger.info("Colonne SQLite ajoutée : %s.%s", table, column)


def _migrate_to_v3(conn) -> None:
    logger.info("Migration SQLite v3 démarrée")
    _ensure_column(conn, "questions", "session_id", "INTEGER")
    _ensure_column(conn, "questions", "chapter_id", "INTEGER")
    _ensure_column(conn, "questions", "question_type", "TEXT")
    _ensure_column(conn, "questions", "choices_json", "TEXT")
    logger.info("Migration SQLite v3 terminée")


def _migrate_to_v4(conn) -> None:
    logger.info("Migration SQLite v4 démarrée")
    renames = (
        ("text_comprehension", "context_comprehension"),
        ("space_vision", "retention"),
        ("math", "curiosity"),
    )
    for old, new in renames:
        _rename_column_if_exists(conn, "metacog_profile", old, new)

    _ensure_column(conn, "metacog_profile", "context_comprehension", "REAL DEFAULT 50.0")
    _ensure_column(conn, "metacog_profile", "retention", "REAL DEFAULT 50.0")
    _ensure_column(conn, "metacog_profile", "curiosity", "REAL DEFAULT 50.0")
    _rename_criterion_values(conn, "metacog_history", "criterion", renames)
    _rename_criterion_values(conn, "session_gauges", "gauge_name", renames)
    logger.info("Migration SQLite v4 terminée")


def _migrate_to_v5(conn) -> None:
    logger.info("Migration SQLite v5 démarrée")
    _ensure_column(conn, "metacog_profile", "meta_cognition", "REAL DEFAULT 50.0")
    logger.info("Migration SQLite v5 terminée")


def _migrate_to_v6(conn) -> None:
    logger.info("Migration SQLite v6 démarrée")
    _ensure_column(conn, "flashcards", "assets_json", "TEXT")
    logger.info("Migration SQLite v6 terminée")


def _migrate_to_v7(conn) -> None:
    logger.info("Migration SQLite v7 démarrée")
    _ensure_column(conn, "user", "speed_ms", "INTEGER DEFAULT 500")
    logger.info("Migration SQLite v7 terminée")


def _migrate_to_v8(conn) -> None:
    logger.info("Migration SQLite v8 démarrée")
    _ensure_column(conn, "documents", "subject", "TEXT")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS subject_profile (
            user_id         INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            subject         TEXT NOT NULL,
            level           REAL DEFAULT 50.0,
            questions_count INTEGER DEFAULT 0,
            correct_count   INTEGER DEFAULT 0,
            updated_at      DATETIME DEFAULT (datetime('now')),
            PRIMARY KEY (user_id, subject)
        );
        CREATE INDEX IF NOT EXISTS idx_subject_profile_user ON subject_profile(user_id);
        """
    )
    logger.info("Migration SQLite v8 terminée")


def _migrate_to_v9(conn) -> None:
    logger.info("Migration SQLite v9 démarrée")
    _ensure_subject_history_table(conn)
    logger.info("Migration SQLite v9 terminée")


def _rename_column_if_exists(conn, table: str, old: str, new: str) -> None:
    columns = _table_columns(conn, table)
    if old not in columns:
        return
    if new in columns:
        logger.warning("Renommage ignoré : %s.%s et %s.%s existent déjà", table, old, table, new)
        return
    conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
    logger.info("Colonne SQLite renommée : %s.%s -> %s", table, old, new)


def _rename_criterion_values(conn, table: str, column: str, renames: tuple[tuple[str, str], ...]) -> None:
    columns = _table_columns(conn, table)
    if column not in columns:
        return
    for old, new in renames:
        conn.execute(f"UPDATE {table} SET {column}=? WHERE {column}=?", (new, old))


def _table_columns(conn, table: str) -> set[str]:
    return {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _ensure_v2_tables(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS user (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            speed_ms    INTEGER DEFAULT 500,
            created_at  DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS answers (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id        INTEGER REFERENCES questions(id) ON DELETE CASCADE,
            user_id            INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            session_id         INTEGER REFERENCES reading_sessions(id) ON DELETE SET NULL,
            answer_text        TEXT NOT NULL,
            verdict            TEXT,
            feedback           TEXT,
            completion         TEXT,
            hint               TEXT,
            response_time_ms   INTEGER,
            length_chars       INTEGER,
            length_words       INTEGER,
            metacog_signals    TEXT,
            attempt_number     INTEGER DEFAULT 1,
            answered_at        DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS session_gauges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES reading_sessions(id) ON DELETE CASCADE,
            t           REAL NOT NULL,
            gauge_name  TEXT NOT NULL,
            value       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metacog_profile (
            user_id             INTEGER PRIMARY KEY REFERENCES user(id) ON DELETE CASCADE,
            context_comprehension REAL DEFAULT 50.0,
            creativity          REAL DEFAULT 50.0,
            retention           REAL DEFAULT 50.0,
            curiosity           REAL DEFAULT 50.0,
            meta_cognition      REAL DEFAULT 50.0,
            attention           REAL DEFAULT 50.0,
            sessions_count      INTEGER DEFAULT 0,
            updated_at          DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS metacog_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            session_id     INTEGER REFERENCES reading_sessions(id) ON DELETE SET NULL,
            criterion      TEXT NOT NULL,
            value_before   REAL NOT NULL,
            value_after    REAL NOT NULL,
            session_score  REAL NOT NULL,
            alpha          REAL NOT NULL,
            recorded_at    DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS flashcards (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            question_id    INTEGER REFERENCES questions(id) ON DELETE SET NULL,
            document_id    INTEGER REFERENCES documents(id) ON DELETE SET NULL,
            chapter_id     INTEGER REFERENCES chapters(id) ON DELETE SET NULL,
            front          TEXT NOT NULL,
            back           TEXT NOT NULL,
            tags           TEXT,
            assets_json    TEXT,
            difficulty     INTEGER DEFAULT 2,
            source         TEXT DEFAULT 'auto',
            last_reviewed  DATETIME,
            review_count   INTEGER DEFAULT 0,
            last_verdict   TEXT,
            created_at     DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rephrasing (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id       INTEGER REFERENCES questions(id) ON DELETE SET NULL,
            session_id        INTEGER REFERENCES reading_sessions(id) ON DELETE SET NULL,
            angle             TEXT,
            rephrased_text    TEXT NOT NULL,
            note              TEXT,
            created_at        DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS session_reflections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER REFERENCES reading_sessions(id) ON DELETE CASCADE,
            user_id         INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            question_text   TEXT NOT NULL,
            answer_text     TEXT NOT NULL,
            question_order  INTEGER DEFAULT 0,
            created_at      DATETIME DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_answers_session ON answers(session_id);
        CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
        CREATE INDEX IF NOT EXISTS idx_session_gauges_session ON session_gauges(session_id);
        CREATE INDEX IF NOT EXISTS idx_flashcards_user ON flashcards(user_id);
        CREATE INDEX IF NOT EXISTS idx_metacog_history_user ON metacog_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_session_reflections_session ON session_reflections(session_id);
        """
    )


def _migrate_to_v10(conn) -> None:
    logger.info("Migration SQLite v10 démarrée")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS login_streak (
            user_id    INTEGER PRIMARY KEY REFERENCES user(id) ON DELETE CASCADE,
            streak     INTEGER DEFAULT 1,
            last_login TEXT NOT NULL
        );
        """
    )
    logger.info("Migration SQLite v10 terminée")


def _migrate_to_v11(conn) -> None:
    logger.info("Migration SQLite v11 démarrée")
    _ensure_column(conn, "questions", "source_context", "TEXT")
    logger.info("Migration SQLite v11 terminée")


def _migrate_to_v12(conn) -> None:
    logger.info("Migration SQLite v12 démarrée")
    _ensure_column(conn, "questions", "source_block_id", "TEXT")
    logger.info("Migration SQLite v12 terminée")


def _migrate_to_v13(conn) -> None:
    logger.info("Migration SQLite v13 démarrée")
    _ensure_column(conn, "pages_cache", "enrich_assets", "INTEGER DEFAULT 1")
    _ensure_column(conn, "pages_cache", "page_plan_json", "TEXT")
    _ensure_column(conn, "pages_cache", "layout_risk_json", "TEXT")
    _ensure_column(conn, "pages_cache", "quality_score", "REAL")
    _ensure_column(conn, "pages_cache", "warnings_json", "TEXT")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS document_index (
            doc_id                   INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
            pdf_hash                 TEXT NOT NULL,
            opendataloader_status    TEXT NOT NULL DEFAULT 'pending',
            detected_document_type   TEXT,
            chapters_json            TEXT,
            global_assets_json       TEXT,
            backend_report_json      TEXT,
            created_at               DATETIME DEFAULT (datetime('now')),
            updated_at               DATETIME DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS llm_pdf_cache (
            cache_key    TEXT NOT NULL,
            task_type    TEXT NOT NULL,
            input_hash   TEXT NOT NULL,
            output_json  TEXT NOT NULL,
            confidence   REAL,
            model        TEXT,
            created_at   DATETIME DEFAULT (datetime('now')),
            PRIMARY KEY (cache_key, task_type)
        );

        CREATE TABLE IF NOT EXISTS asset_cache (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number  INTEGER NOT NULL,
            block_id     TEXT,
            asset_type   TEXT NOT NULL,
            image_path   TEXT NOT NULL,
            bbox         TEXT,
            source       TEXT,
            confidence   REAL,
            created_at   DATETIME DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_document_index_hash ON document_index(pdf_hash);
        CREATE INDEX IF NOT EXISTS idx_llm_pdf_cache_task ON llm_pdf_cache(task_type);
        CREATE INDEX IF NOT EXISTS idx_asset_cache_doc_page ON asset_cache(doc_id, page_number);
        """
    )
    logger.info("Migration SQLite v13 terminée")


def _migrate_to_v14(conn) -> None:
    logger.info("Migration SQLite v14 démarrée")
    _ensure_column(conn, "user", "lang", "TEXT NOT NULL DEFAULT 'fr'")
    logger.info("Migration SQLite v14 terminée")


def _ensure_subject_history_table(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS subject_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            session_id     INTEGER REFERENCES reading_sessions(id) ON DELETE SET NULL,
            subject        TEXT NOT NULL,
            value_before   REAL NOT NULL,
            value_after    REAL NOT NULL,
            source         TEXT DEFAULT 'session',
            recorded_at    DATETIME DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_subject_history_user ON subject_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_subject_history_subject ON subject_history(user_id, subject);
        """
    )
