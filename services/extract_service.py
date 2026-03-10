from __future__ import annotations

import json
import re

from pydantic import ValidationError

from schemas.models import ExtractionInput, ResumeStructured
from services.llm_service import call_llm
from utils.errors import LLMParseError, NotResumeError


RESUME_HINT_KEYWORDS = {
    "education",
    "experience",
    "project",
    "projects",
    "skills",
    "summary",
    "work experience",
    "employment",
    "university",
    "college",
    "bachelor",
    "master",
    "intern",
    "research",
}


def _normalize_text(text: str) -> str:
    """Normalize whitespace for downstream checks."""
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_resume(text: str) -> bool:
    """Apply a small heuristic check before calling the LLM."""
    normalized = _normalize_text(text).lower()

    if len(normalized) < 80:
        return False

    hit_count = sum(1 for kw in RESUME_HINT_KEYWORDS if kw in normalized)
    return hit_count >= 2


def _extract_json(raw_output: str) -> dict:
    """Extract JSON object from raw LLM output."""
    raw_output = raw_output.strip()

    fenced_match = re.search(r"```json\s*(\{.*\})\s*```", raw_output, flags=re.DOTALL)
    if fenced_match:
        raw_output = fenced_match.group(1).strip()

    generic_match = re.search(r"```\s*(\{.*\})\s*```", raw_output, flags=re.DOTALL)
    if generic_match:
        raw_output = generic_match.group(1).strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise LLMParseError("LLM output is not valid JSON") from exc


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


def extract_structured_resume(data: ExtractionInput) -> ResumeStructured:
    """Convert raw resume text into a validated ResumeStructured object."""
    if not data.text.strip():
        raise NotResumeError("Input text is empty")

    if not _looks_like_resume(data.text):
        raise NotResumeError("Input text does not look like a resume")

    prompt = _build_prompt(data.text)
    raw_output = call_llm(prompt)
    parsed = _extract_json(raw_output)

    try:
        return ResumeStructured.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(f"LLM output does not match ResumeStructured schema: {exc}") from exc