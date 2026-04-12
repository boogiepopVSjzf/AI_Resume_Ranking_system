from __future__ import annotations

from storage.s3_storage import build_resume_storage_key


def test_build_resume_storage_key_normalizes_extension():
    assert build_resume_storage_key("abc123", "pdf") == "raw/abc123.pdf"


def test_build_resume_storage_key_preserves_prefixed_extension():
    assert build_resume_storage_key("abc123", ".docx") == "raw/abc123.docx"
