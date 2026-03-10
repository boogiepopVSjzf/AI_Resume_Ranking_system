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

class PDFParseError(AppError):
    """Raised when PDF parsing fails."""
    pass

class LLMParseError(AppError):
    """Raised when LLM response cannot be parsed or validated."""
    # 对应 llm_service.py 中：raise LLMParseError("Invalid LLM API response format")
    pass

class NotResumeError(AppError):
    """Raised when input text does not look like a resume."""
    pass

class NotResumeError(AppError):
    """Raised when input text does not look like a resume."""
    pass