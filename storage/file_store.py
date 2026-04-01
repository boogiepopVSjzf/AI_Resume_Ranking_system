from __future__ import annotations

import uuid
from pathlib import Path

from config import settings


def ensure_storage_dirs() -> None:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.TXT_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def new_resume_id() -> str:
    return uuid.uuid4().hex


def upload_stored_path(resume_id: str, ext: str) -> Path:
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return settings.UPLOAD_DIR / f"{resume_id}{ext}"


def save_upload_bytes(resume_id: str, ext: str, content: bytes) -> Path:
    ensure_storage_dirs()
    path = upload_stored_path(resume_id, ext)
    path.write_bytes(content)
    return path


def txt_path(resume_id: str) -> Path:
    return settings.TXT_DIR / f"{resume_id}.txt"


def save_txt(resume_id: str, text: str) -> Path:
    ensure_storage_dirs()
    path = txt_path(resume_id)
    path.write_text(text, encoding="utf-8")
    return path


def result_path(resume_id: str) -> Path:
    return settings.RESULTS_DIR / f"{resume_id}.json"


def save_result_json(resume_id: str, json_text: str) -> Path:
    ensure_storage_dirs()
    path = result_path(resume_id)
    path.write_text(json_text, encoding="utf-8")
    return path
