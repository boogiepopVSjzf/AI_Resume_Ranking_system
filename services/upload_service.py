"""
Upload service layer for handling file upload business logic.
Keeps API routes clean by encapsulating complex operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import settings
from services.document_to_txt import extract_text_from_document
from services.document_validate import allowed_types_hint, validate_upload_magic
from storage.file_store import new_resume_id, save_upload_bytes, save_txt
from utils.constants import ERR_UNSUPPORTED_FILE_TYPE
from utils.errors import (
    CorruptedPDFError,
    DocumentExtractError,
    EncryptedPDFError,
    FileSizeError,
    InvalidFileType,
    InvalidResumeError,
)
from utils.logger import get_logger


logger = get_logger("upload_service")


@dataclass
class UploadResult:
    """Result of a successful upload operation."""
    resume_id: str
    text: str
    txt_path: str


@dataclass
class BatchUploadResult:
    """Result of a batch upload operation."""
    total: int
    succeeded_count: int
    failed_count: int
    succeeded: list[dict]
    failed: list[dict]


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


def process_upload(ext: str, content: bytes) -> UploadResult:
    """
    Process uploaded file content: validate, save, convert to text.
    
    Args:
        ext: File extension (e.g., '.pdf', '.docx')
        content: File content bytes
        
    Returns:
        UploadResult with resume_id, text, and txt_path
        
    Raises:
        InvalidFileType: If file content doesn't match extension
        FileSizeError: If file is too small/large
        EncryptedPDFError: If PDF is encrypted
        CorruptedPDFError: If PDF is corrupted
        DocumentExtractError: If text extraction fails
    """
    validate_upload_magic(ext, content)
    
    resume_id = new_resume_id()
    upload_path = save_upload_bytes(resume_id, ext, content)
    
    try:
        text = extract_text_from_document(upload_path)
    except Exception:
        # Clean up uploaded file on any parsing failure
        _safe_unlink(upload_path)
        raise

    # Validate that the extracted text is a valid resume
    checker = ResumeValidityChecker()
    validity_result = checker.check_text(text)
    if validity_result.decision == "HARD_FAIL":
        _safe_unlink(upload_path)
        raise InvalidResumeError("上传的文件似乎不是一份有效的简历")

    
    txt_path = save_txt(resume_id, text)
    logger.info("Processed upload: resume_id=%s, txt=%s", resume_id, txt_path.name)
    
    return UploadResult(
        resume_id=resume_id,
        text=text,
        txt_path=f"storage/txts/{txt_path.name}",
    )


def process_single_file_in_batch(
    filename: str,
    ext: str,
    content: bytes,
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Process a single file in batch upload, reusing the core process_upload logic.
    
    Returns:
        Tuple of (success_dict, failure_dict) - one will be None
    """
    try:
        result = process_upload(ext, content)
        logger.info("[BATCH] Processed %s -> resume_id=%s", filename, result.resume_id)
        return {
            "resume_id": result.resume_id,
            "filename": filename,
            "txt_path": result.txt_path,
        }, None
    except (InvalidFileType, FileSizeError, EncryptedPDFError, CorruptedPDFError, DocumentExtractError) as exc:
        logger.error("[BATCH] %s: %s", filename, exc)
        return None, {"filename": filename, "reason": str(exc)}
    except Exception as exc:
        logger.error("[BATCH] Unexpected error for %s: %s", filename, exc)
        return None, {"filename": filename, "reason": f"处理失败: {exc}"}


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
