from __future__ import annotations

import json
from typing import Any

from storage.db_config import get_connection


def save_parsed_resume(resume_id: str, structured: Any) -> None:
    if hasattr(structured, "model_dump_json"):
        profile_json = structured.model_dump_json(ensure_ascii=False)
    elif isinstance(structured, dict):
        profile_json = json.dumps(structured, ensure_ascii=False)
    else:
        raise TypeError("structured must be a Pydantic model or dict")

    sql = """
    INSERT OR REPLACE INTO resumes (resume_id, profile_json)
    VALUES (?, ?)
    """

    with get_connection() as conn:
        conn.execute(sql, (resume_id, profile_json))
        conn.commit()
