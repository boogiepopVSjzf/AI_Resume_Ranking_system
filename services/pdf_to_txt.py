from pathlib import Path

from pypdf import PdfReader

from config import settings
from services.text_clean_service import clean_text
from utils.errors import PDFParseError


def extract_raw_text(pdf_path: Path) -> str:
    # Note: extract_text can be unreliable for multi-column or table PDFs; consider another parser if accuracy is an issue.
    try:
        reader = PdfReader(str(pdf_path))
        if getattr(reader, "is_encrypted", False):
            raise PDFParseError("PDF 已加密，无法解析")
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return "\n".join(parts)
    except PDFParseError:
        raise
    except Exception as exc:
        raise PDFParseError("PDF 解析失败") from exc


def pdf_to_txt(pdf_path: Path) -> str:
    raw = extract_raw_text(pdf_path)
    text = clean_text(raw)
    if len("".join(text.split())) < settings.MIN_EXTRACTED_TEXT_CHARS:
        raise PDFParseError("PDF 不包含可提取的文本（疑似图片 PDF）")
    return text
