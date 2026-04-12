from __future__ import annotations

import re
from typing import Optional

from pydantic import ValidationError

from schemas.models import ExtractionInput, ResumeStructured
from utils.errors import LLMParseError, NotResumeError
from services.llm_service import call_llm
from utils.llm_json import extract_json


def _normalize_text(text: str) -> str:
    """Normalize whitespace for downstream checks."""
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_resume(text: str) -> bool:
    normalized = _normalize_text(text)
    if len(normalized) < 40:
        return False
    lowered = normalized.lower()
    score = 0
    indicators = (
        "experience",
        "education",
        "skills",
        "projects",
        "work experience",
        "professional experience",
    )
    for indicator in indicators:
        if indicator in lowered:
            score += 1
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", normalized):
        score += 1
    if re.search(r"\+?\d[\d()\-\s]{7,}\d", normalized):
        score += 1
    return score >= 2

#根据 ResumeStructured 模型的 JSON schema 构建提取提示,要改prompt也是在这里改
def _build_prompt(text: str) -> str:
    """Build a concise extraction prompt to reduce token cost and latency."""
    return f"""
You are a resume information extraction system.

Extract structured resume information from the input text.

Rules:
1. Return JSON only.
2. Do not wrap the JSON in markdown.
3. Do not invent information that is not explicitly supported by the text.
4. If a field is missing, use null for scalar fields and [] for list fields.
5. This is an English-only system. Use English enum values exactly as required by the schema.
6. For education_level, choose exactly one of: "high_school", "associate", "bachelor", "master", "phd", "other".
7. For major, choose exactly one of: "computer_science", "mathematics", "medicine", "finance", "engineering", "other".
8. Keep education[].major as the original major text from the resume when present. Use the top-level `major` field for the standardized category.
9. Follow this exact JSON shape:
{{
  "name": string | null,
  "email": string | null,
  "phone": string | null,
  "YoE": string | null,
  "education_level": "high_school" | "associate" | "bachelor" | "master" | "phd" | "other" | null,
  "major": "computer_science" | "mathematics" | "medicine" | "finance" | "engineering" | "other" | null,
  "location": string | null,
  "skills": string[],
  "education": [
    {{
      "school": string | null,
      "degree": string | null,
      "major": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "description": string | null
    }}
  ],
  "experience": [
    {{
      "company": string | null,
      "title": string | null,
      "location": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "highlights": string[]
    }}
  ],
  "projects": [
    {{
      "name": string | null,
      "role": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "highlights": string[]
    }}
  ]
}}

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

    if not _looks_like_resume(data.text):
        raise NotResumeError("Input text does not look like a resume")

    prompt = _build_prompt(data.text)
    raw_output, usage = call_llm(prompt, provider=provider, model=model)
    parsed = extract_json(raw_output)

    try:
        return ResumeStructured.model_validate(parsed), usage
    except ValidationError as exc:
        raise LLMParseError(
            f"LLM output does not match ResumeStructured schema: {exc}"
        ) from exc
