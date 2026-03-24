import io
from pathlib import Path

import pytest
from docx import Document

from config import settings
from services.document_to_txt import document_to_txt
from utils.errors import DocumentExtractError, FileSizeError


def _minimal_docx_bytes() -> bytes:
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("x" * max(settings.MIN_EXTRACTED_TEXT_CHARS + 10, 50))
    doc.save(buf)
    return buf.getvalue()


def test_document_to_txt_docx_roundtrip(tmp_path: Path):
    path = tmp_path / "r.docx"
    path.write_bytes(_minimal_docx_bytes())
    text = document_to_txt(path)
    assert len(text.strip()) >= settings.MIN_EXTRACTED_TEXT_CHARS


def test_document_to_txt_docx_empty_raises(tmp_path: Path):
    buf = io.BytesIO()
    doc = Document()
    doc.save(buf)
    path = tmp_path / "empty.docx"
    path.write_bytes(buf.getvalue())
    with pytest.raises(DocumentExtractError):
        document_to_txt(path)


def test_document_to_txt_rejects_tiny_file_on_disk(tmp_path: Path):
    path = tmp_path / "tiny.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(FileSizeError):
        document_to_txt(path)
