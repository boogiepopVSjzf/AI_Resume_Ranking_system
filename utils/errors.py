from __future__ import annotations


class AppError(Exception):
    """Base application error."""
    pass


class InvalidFileType(AppError):
    """Raised when uploaded file type is not supported."""
    pass


class PDFParseError(AppError):
    """Raised when PDF parsing fails."""
    pass


class LLMParseError(AppError):
    """Raised when LLM response cannot be parsed or validated."""
    pass


class NotResumeError(AppError):
    """Raised when input text does not look like a resume."""
    pass
