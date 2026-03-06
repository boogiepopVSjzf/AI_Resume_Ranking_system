class AppError(Exception):
    pass


class InvalidFileType(AppError):
    pass


class FileSizeError(AppError):
    """File is outside the accepted size range (too small or too large)."""
    pass


class PDFParseError(AppError):
    pass


class EncryptedPDFError(PDFParseError):
    """PDF is password-protected and cannot be read."""
    pass


class CorruptedPDFError(PDFParseError):
    """PDF structure is damaged or invalid and cannot be parsed."""
    pass


class LLMParseError(AppError):
    pass
