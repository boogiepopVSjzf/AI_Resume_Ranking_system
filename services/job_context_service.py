from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from services.document_validate import validate_upload_magic
from services.pdf_to_txt import extract_text_from_pdf
from utils.logger import get_logger

logger = get_logger("job_context")


class JDSourceConflict(ValueError):
    """jd_text and jd_file provided simultaneously."""


class JobContextEmpty(ValueError):
    """Neither hr_note nor JD body has content."""


def jd_text_from_pdf(content: bytes) -> str:
    """Extract plain text from in-memory PDF bytes.

    Writes to a temp file so that the existing extract_text_from_pdf pipeline
    (which expects a Path) can be reused without modification.
    """
    validate_upload_magic(".pdf", content)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(content)
        tmp.flush()
        tmp.close()
        return extract_text_from_pdf(Path(tmp.name))
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def resolve_jd_body(
    jd_text: Optional[str],
    jd_file_content: Optional[bytes],
) -> str:
    """Return the resolved JD body text.

    Raises JDSourceConflict when both channels carry content.
    """
    has_text = bool(jd_text and jd_text.strip())
    has_file = bool(jd_file_content)

    if has_text and has_file:
        raise JDSourceConflict("JD 请通过纯文本或 PDF 二选一提供，不能同时传入")

    if has_file:
        return jd_text_from_pdf(jd_file_content)
    if has_text:
        return jd_text.strip()
    return ""


def build_job_context(hr_note: str, jd_body: str) -> dict[str, Any]:
    """Merge hr_note and jd_body into a structured response dict.

    Raises JobContextEmpty if both are blank.
    """
    hr_note = hr_note.strip()
    jd_body = jd_body.strip()

    if not hr_note and not jd_body:
        raise JobContextEmpty("hr_note 与 JD 至少需要提供一项")

    parts: list[str] = []
    if hr_note:
        parts.append(f"HR_NOTE:\n{hr_note}")
    if jd_body:
        parts.append(f"JOB_DESCRIPTION:\n{jd_body}")

    return {
        "hr_note": hr_note,
        "jd_text": jd_body,
        "merged_context": "\n\n".join(parts),
    }
