class AppError(Exception):
    pass


class InvalidFileType(AppError):
    pass


class PDFParseError(AppError):
    pass


class LLMParseError(AppError):
    pass
