from __future__ import annotations

import sqlite3
from pathlib import Path

from config import settings
from schemas.models import ResumeStructured


# Path to the SQLite database file (stored under the existing storage directory)
DB_PATH: Path = settings.STORAGE_DIR / "resume_ranking.db"


def get_connection() -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.

    If the parent directory does not exist, it will be created.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db() -> None:
    """
    Initialize the database schema for storing parsed resumes.

    v1: we only create a single table 'resumes' to store
    the structured resume JSON.

    Later we can extend this schema (add more tables or fields)
    for ranking, jobs, search_results, etc.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            resume_id    TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    conn.commit()
    conn.close()


def save_parsed_resume(resume_id: str, structured: ResumeStructured) -> None:
    """
    Save a parsed structured resume into SQLite.

    Corner cases handled here:
    - empty or invalid resume_id
    - database/table not initialized yet
    - repeated writes to the same resume_id
    - rollback on database write failure
    """
    if not isinstance(resume_id, str) or not resume_id.strip():
        raise ValueError("resume_id must be a non-empty string")

    # Ensure the database and table exist before writing
    init_db()

    conn = get_connection()
    try:
        cur = conn.cursor()

        json_text = structured.model_dump_json(ensure_ascii=False)

        cur.execute(
            """
            INSERT OR REPLACE INTO resumes (resume_id, profile_json)
            VALUES (?, ?);
            """,
            (resume_id, json_text),
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
