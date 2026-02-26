from __future__ import annotations

import uuid
from pathlib import Path

from config import settings


def ensure_storage_dirs() -> None:
    settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
    settings.TXT_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def new_resume_id() -> str:
    return uuid.uuid4().hex


def pdf_path(resume_id: str) -> Path:
    return settings.PDF_DIR / f"{resume_id}.pdf"


def txt_path(resume_id: str) -> Path:
    return settings.TXT_DIR / f"{resume_id}.txt"


def save_pdf_bytes(resume_id: str, pdf_bytes: bytes) -> Path:
    ensure_storage_dirs()
    path = pdf_path(resume_id)
    path.write_bytes(pdf_bytes)
    return path


def save_txt(resume_id: str, text: str) -> Path:
    ensure_storage_dirs()
    path = txt_path(resume_id)
    path.write_text(text, encoding="utf-8")
    return path
