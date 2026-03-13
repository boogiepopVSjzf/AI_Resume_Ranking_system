from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from config import settings
from services.text_clean_service import clean_text
from utils.errors import CorruptedPDFError, EncryptedPDFError, FileSizeError, PDFParseError
from utils.logger import get_logger

logger = get_logger("pdf_to_txt")

# Heuristic thresholds for detecting multi-column layout:
# if the average non-empty line length is below this value AND there are
# at least this many lines, the text is likely column-scrambled.
_MULTICOLUMN_AVG_LINE_LEN = 40
_MULTICOLUMN_MIN_LINES = 20


def _validate_size(pdf_path: Path) -> None:
    size = pdf_path.stat().st_size
    if size < settings.MIN_UPLOAD_BYTES:
        logger.warning("File too small, skipping: %s (%d bytes)", pdf_path.name, size)
        raise FileSizeError(f"File too small: {pdf_path.name} ({size} bytes)")
    if size > settings.MAX_UPLOAD_BYTES:
        logger.warning("File too large, skipping: %s (%d bytes)", pdf_path.name, size)
        raise FileSizeError(f"File too large: {pdf_path.name} ({size} bytes)")


def _looks_multicolumn(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < _MULTICOLUMN_MIN_LINES:
        return False
    return (sum(len(line) for line in lines) / len(lines)) < _MULTICOLUMN_AVG_LINE_LEN


def extract_raw_text(pdf_path: Path) -> str:
    # Note: standard extract_text can scramble multi-column layouts; layout-mode fallback is
    # applied automatically when the heuristic detects suspiciously short average line lengths.
    try:
        reader = PdfReader(str(pdf_path))
        if getattr(reader, "is_encrypted", False):
            raise PDFParseError("PDF 已加密，无法解析")
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        text = "\n".join(parts)
    except FileNotDecryptedError as exc:
        logger.error("Encrypted PDF, skipping: %s", pdf_path.name)
        raise EncryptedPDFError(f"PDF is encrypted: {pdf_path.name}") from exc
    except PdfReadError as exc:
        logger.error("Corrupted PDF, skipping: %s — %s", pdf_path.name, exc)
        raise CorruptedPDFError(f"PDF is corrupted: {pdf_path.name}") from exc
    except PDFParseError:
        raise
    except Exception as exc:
        logger.error("Unexpected parse error, skipping: %s — %s", pdf_path.name, exc)
        raise PDFParseError(f"Failed to parse PDF: {pdf_path.name}") from exc

    if not text.strip():
        logger.warning("PDF yielded no extractable text: %s", pdf_path.name)
        raise PDFParseError(f"PDF contains no extractable text: {pdf_path.name}")

    if _looks_multicolumn(text):
        logger.warning(
            "Multi-column layout detected in %s — retrying with layout extraction mode; "
            "accuracy may be reduced",
            pdf_path.name,
        )
        try:
            reader2 = PdfReader(str(pdf_path))
            layout_parts = []
            for page in reader2.pages:
                layout_parts.append(page.extract_text(extraction_mode="layout") or "")
            layout_text = "\n".join(layout_parts)
            if len(layout_text) > len(text):
                text = layout_text
        except Exception as exc:
            logger.warning(
                "Layout-mode fallback also failed for %s — %s", pdf_path.name, exc
            )

    return text


def pdf_to_txt(pdf_path: Path) -> str:
    _validate_size(pdf_path)
    raw = extract_raw_text(pdf_path)
    text = clean_text(raw)
    if len("".join(text.split())) < settings.MIN_EXTRACTED_TEXT_CHARS:
        raise PDFParseError("PDF 不包含可提取的文本（疑似图片 PDF）")
    return text
