"""
Upload service layer for handling file upload business logic.
Keeps API routes clean by encapsulating complex operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from config import settings
from services.document_to_txt import extract_text_from_document
from services.document_validate import allowed_types_hint, validate_upload_magic
from utils.constants import ERR_UNSUPPORTED_FILE_TYPE
from utils.errors import (
    CorruptedPDFError,
    DocumentExtractError,
    EncryptedPDFError,
    FileSizeError,
    InvalidFileType,
)
from utils.logger import get_logger


logger = get_logger("upload_service")


@dataclass
class UploadResult:
    """Result of a successful upload operation."""
    resume_id: str
    text: str


def validate_filename(filename: Optional[str]) -> str:
    """Validate filename and return the extension."""
    if not filename:
        raise InvalidFileType("未选择文件")
    if len(filename) > settings.MAX_FILENAME_LENGTH:
        raise InvalidFileType("文件名过长")
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileType(f"仅支持 {allowed_types_hint()} 文件")
    return ext


def process_upload_with_resume_id(ext: str, content: bytes, resume_id: str) -> UploadResult:
    """
    Process uploaded file content with a caller-supplied resume_id.
    This lets upstream services parallelise storage upload and text extraction.
    """
    upload_path: Optional[Path] = None

    try:
        with NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            upload_path = Path(tmp.name)
        text = extract_text_from_document(upload_path)
    except Exception:
        if upload_path is not None:
            _safe_unlink(upload_path)
        raise
    finally:
        if upload_path is not None:
            _safe_unlink(upload_path)

    logger.info("Processed upload: resume_id=%s", resume_id)
    
    return UploadResult(
        resume_id=resume_id,
        text=text,
    )

def validate_batch_file(filename: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    """
    Validate a file in batch upload.
    
    Returns:
        Tuple of (extension, display_filename, failure_dict)
        - If valid: (ext, filename, None)
        - If invalid: (None, display_filename, failure_dict)
    """
    display_name = filename or "<unknown>"
    
    if not filename:
        return None, display_name, {"filename": display_name, "reason": "未提供文件名"}
    
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return None, display_name, {"filename": display_name, "reason": f"{ERR_UNSUPPORTED_FILE_TYPE}: {ext}"}
    
    return ext, display_name, None


def _safe_unlink(path: Path) -> None:
    """Safely delete a file, ignoring errors."""
    try:
        path.unlink()
    except Exception:
        pass
