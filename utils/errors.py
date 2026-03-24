from __future__ import annotations

class AppError(Exception):
    """Base application error."""
    # 增加构造函数，以支持 llm_service 中可能需要的错误详情传递
    def __init__(self, message: str, *, code: str = "APP_ERROR", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

class LLMError(AppError):
    """Raised when LLM call fails (e.g., Auth, Timeout, or Connection errors)."""
    # 对应 llm_service.py 中：raise LLMError(f"Permanent LLM failure: {str(e)}")
    def __init__(self, message: str, *, code: str = "LLM_ERROR", details: dict | None = None):
        super().__init__(message, code=code, details=details)

class InvalidFileType(AppError):
    """Raised when uploaded file type is not supported."""
    pass


class FileSizeError(AppError):
    """File is outside the accepted size range (too small or too large)."""
    pass


class DocumentExtractError(AppError):
    """Raised when text cannot be extracted from an uploaded document (non-PDF-specific)."""
    pass


class PDFParseError(DocumentExtractError):
    """Raised when PDF parsing or text extraction fails."""
    pass


class EncryptedPDFError(PDFParseError):
    """PDF is password-protected and cannot be read."""
    pass


class CorruptedPDFError(PDFParseError):
    """PDF structure is damaged or invalid and cannot be parsed."""
    pass


class LLMParseError(AppError):
    """Raised when LLM response cannot be parsed or validated."""
    pass


class NotResumeError(AppError):
    """Raised when input text does not look like a resume."""
    pass