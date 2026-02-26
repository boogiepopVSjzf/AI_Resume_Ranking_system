from pathlib import Path

from pypdf import PdfReader

from services.text_clean_service import clean_text
from utils.errors import PDFParseError


def pdf_to_text(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return clean_text("\n".join(parts))
    except Exception as exc:
        raise PDFParseError("PDF 解析失败") from exc
