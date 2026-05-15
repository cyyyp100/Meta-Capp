# db/__init__.py — Connexion SQLite par thread (threading.local)
import sqlite3
import logging
import threading
from pathlib import Path
from config.settings import DB_PATH

logger = logging.getLogger("DB")
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
        logger.info("Connexion SQLite ouverte (thread=%s) : %s", threading.current_thread().name, DB_PATH)
    return conn


def close_connection() -> None:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn:
        conn.close()
        _local.conn = None
        logger.info("Connexion SQLite fermée (thread=%s).", threading.current_thread().name)
