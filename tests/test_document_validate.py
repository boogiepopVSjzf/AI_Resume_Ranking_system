import io
import zipfile

import pytest

from services.document_validate import (
    allowed_types_hint,
    validate_upload_magic,
)
from utils.errors import InvalidFileType


def test_allowed_types_hint_lists_extensions():
    hint = allowed_types_hint()
    assert "PDF" in hint
    assert "DOCX" in hint


def test_validate_upload_magic_pdf_ok():
    validate_upload_magic(".pdf", b"%PDF-1.4\n%EOF")


def test_validate_upload_magic_pdf_rejects_non_pdf():
    with pytest.raises(InvalidFileType):
        validate_upload_magic(".pdf", b"not a pdf header")


def test_validate_upload_magic_docx_ok():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document></w:document>")
    validate_upload_magic(".docx", buf.getvalue())


def test_validate_upload_magic_docx_rejects_plain_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises(InvalidFileType):
        validate_upload_magic(".docx", buf.getvalue())


def test_validate_upload_magic_docx_rejects_non_zip():
    with pytest.raises(InvalidFileType):
        validate_upload_magic(".docx", b"notzip")
