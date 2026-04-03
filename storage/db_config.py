from __future__ import annotations

import sqlite3
from pathlib import Path

from config import settings

DB_PATH = settings.STORAGE_DIR / "resume_ranking.db"
SQL_DIR = Path(__file__).resolve().parent / "sql"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _read_sql(filename: str) -> str:
    path = SQL_DIR / filename
    return path.read_text(encoding="utf-8")


def init_db() -> None:
    create_sql = _read_sql("create_resumes.sql")
    with get_connection() as conn:
        conn.executescript(create_sql)
        conn.commit()
