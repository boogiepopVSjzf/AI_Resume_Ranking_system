from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from config import settings
from utils.errors import FileSizeError, InvalidFileType
from utils.logger import get_logger

logger = get_logger("document_validate")


def allowed_types_hint() -> str:
    order = [".pdf", ".docx"]
    labels: list[str] = []
    for ext in order:
        if ext in settings.ALLOWED_EXTENSIONS:
            labels.append(ext.upper().lstrip("."))
    for ext in sorted(settings.ALLOWED_EXTENSIONS):
        if ext not in order:
            labels.append(ext.upper().lstrip("."))
    return "、".join(labels)


def validate_file_size(path: Path) -> None:
    size = path.stat().st_size
    if size < settings.MIN_UPLOAD_BYTES:
        logger.warning("File too small, skipping: %s (%d bytes)", path.name, size)
        raise FileSizeError(f"File too small: {path.name} ({size} bytes)")
    if size > settings.MAX_UPLOAD_BYTES:
        logger.warning("File too large, skipping: %s (%d bytes)", path.name, size)
        raise FileSizeError(f"File too large: {path.name} ({size} bytes)")


def _zip_has_word_document(content: bytes) -> bool:
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for name in zf.namelist():
                normalized = name.replace("\\", "/").lower()
                if normalized == "word/document.xml":
                    return True
        return False
    except zipfile.BadZipFile:
        return False


def validate_upload_magic(ext: str, content: bytes) -> None:
    ext = ext.lower()
    hint = allowed_types_hint()
    if ext == ".pdf":
        head = content[:1024].lstrip()
        if not head.startswith(b"%PDF-"):
            raise InvalidFileType(f"文件内容与扩展名不符，仅支持 {hint}")
    elif ext == ".docx":
        if len(content) < 4 or not content.startswith(b"PK"):
            raise InvalidFileType(f"不是有效的 DOCX 文件，仅支持 {hint}")
        if not _zip_has_word_document(content):
            raise InvalidFileType(f"不是有效的 DOCX 文件（缺少文档主体），仅支持 {hint}")
    else:
        raise InvalidFileType(f"不支持的文件类型，仅支持 {hint}")
