from pathlib import Path

from pypdf import PdfReader

from services.text_clean_service import clean_text
from utils.errors import PDFParseError


def extract_raw_text(pdf_path: Path) -> str:
    # Note: extract_text can be unreliable for multi-column or table PDFs; consider another parser if accuracy is an issue.
    try:
        reader = PdfReader(str(pdf_path))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception as exc:
        raise PDFParseError("PDF 解析失败") from exc


def pdf_to_txt(pdf_path: Path) -> str:
    raw = extract_raw_text(pdf_path)
    return clean_text(raw)
