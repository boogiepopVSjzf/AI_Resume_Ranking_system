import re
from typing import Literal

from config import settings
from utils.errors import DocumentExtractError, PDFParseError

#处理空行，使文本更加规整，空白稳定，让后续llm处理不容易呗奇怪字符所干扰
_SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
_BULLET_RE = re.compile(r"^[\s]*[•·●◦▪▫–—\-]+[\s]+", re.MULTILINE)
_DASHES_RE = re.compile(r"[‐‑‒–—−]")


def clean_text(raw: str) -> str:
    if not raw:
        return ""

    text = raw
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n")
    text = _DASHES_RE.sub("-", text)

    text = _BULLET_RE.sub("- ", text)

    lines = [line.strip() for line in text.split("\n")]
    compact_lines = []
    prev_empty = False
    for line in lines:
        if line:
            compact_lines.append(line)
            prev_empty = False
            continue
        if not prev_empty:
            compact_lines.append("")
            prev_empty = True
    text = "\n".join(compact_lines)

    text = _SPACE_RE.sub(" ", text)

    text = text.strip()

    return text


def finalize_extracted_plaintext(raw: str, *, source: Literal["pdf", "docx"]) -> str:
    text = clean_text(raw)
    if len("".join(text.split())) < settings.MIN_EXTRACTED_TEXT_CHARS:
        if source == "pdf":
            raise PDFParseError("PDF 不包含可提取的文本（疑似图片 PDF）")
        raise DocumentExtractError("文档不含足够可提取的文本（可能几乎为空或主要为图片）")
    return text
