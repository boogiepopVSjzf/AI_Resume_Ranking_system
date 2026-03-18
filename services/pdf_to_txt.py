from pathlib import Path

from pypdf import PdfReader

from config import settings
from services.text_clean_service import clean_text
from utils.errors import PDFParseError


def extract_raw_text(pdf_path: Path) -> str: #前端网页先把pdf读进来，再调用这个函数做txt转换
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
#之前在测试的时候，会出现读一个pdf生成的txt只有一行但是特别长的情况，这是由于pdf文件的格式导致的
#其实这无伤大雅，只需要最后llm能正常提取这些txt并且把它们转换成对应的json 结构化数据即可