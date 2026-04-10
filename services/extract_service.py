from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import ValidationError

from schemas.models import ExtractionInput, ResumeStructured
from services.llm_service import call_llm
from utils.errors import LLMParseError, NotResumeError
from utils.llm_json import extract_json as _extract_json


def _normalize_text(text: str) -> str:
    """Normalize whitespace for downstream checks."""
    return re.sub(r"\s+", " ", text).strip()


def _build_resume_check_prompt(text: str) -> str:
    return f"""
You are a resume classification system.

Decide whether the input text is a resume/CV.

Return JSON only. Do not wrap in markdown.

JSON schema:
{{
  "is_resume": boolean
}}

Text:
{text}
""".strip()


def _looks_like_resume(text: str, provider: Optional[str] = None, model: Optional[str] = None) -> bool:
    normalized = _normalize_text(text)
    if len(normalized) < 40:
        return False

    snippet = normalized[:4000]
    prompt = _build_resume_check_prompt(snippet)
    raw_output, _ = call_llm(prompt, provider=provider, model=model)
    parsed = _extract_json(raw_output)

    value = parsed.get("is_resume")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False

    raise LLMParseError("LLM output missing boolean field: is_resume")

#根据 ResumeStructured 模型的 JSON schema 构建提取提示,要改prompt也是在这里改
def _build_prompt(text: str) -> str:
    """Build the extraction prompt using the schema contract."""
    schema_dict = ResumeStructured.model_json_schema()

    return f"""
You are a resume information extraction system.

Extract structured resume information from the input text.

Rules:
1. Return JSON only.
2. Do not wrap the JSON in markdown.
3. Do not invent information that is not explicitly supported by the text.
4. If a field is missing, use null for scalar fields and [] for list fields.
5. Follow this JSON schema exactly:

{json.dumps(schema_dict, ensure_ascii=False, indent=2)}

Resume text:
{text}
""".strip()

#整合前面的函数来实现从原始文本到结构化简历的提取
def extract_structured_resume(
    data: ExtractionInput,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[ResumeStructured, dict]:
    """
    Convert raw resume text into a validated ResumeStructured object.
    """
    if not data.text.strip():
        raise NotResumeError("Input text is empty")

    if not _looks_like_resume(data.text, provider=provider, model=model):
        raise NotResumeError("Input text does not look like a resume")

    prompt = _build_prompt(data.text)
    raw_output, usage = call_llm(prompt, provider=provider, model=model)
    parsed = _extract_json(raw_output)

    try:
        return ResumeStructured.model_validate(parsed), usage
    except ValidationError as exc:
        raise LLMParseError(
            f"LLM output does not match ResumeStructured schema: {exc}"
        ) from exc
