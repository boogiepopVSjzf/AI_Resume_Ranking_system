from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from schemas.job_query import StandardizedJobQuery
from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json


def _build_rewrite_prompt(merged_context: str) -> str:
    schema_dict = StandardizedJobQuery.model_json_schema()

    return f"""
You are a job-requirement analysis system.

Given the merged context below (which may contain an HR note and/or a job description),
produce a structured JSON object that decomposes the requirements into two parts:

1. **hard_filters** — deterministic filtering criteria:
   - min_yoe (integer or null): minimum years of experience required.
   - required_skills (list of strings): key skills explicitly mentioned; preserve the original language; deduplicate.
   - education (string or null): minimum education level required (e.g. "本科", "硕士", "Bachelor", "Master").
   - location (string or null): required work location if mentioned.
2. **search_query** — a concise natural-language sentence (one or two sentences) summarising the ideal candidate profile for semantic search. Do NOT just copy the input; distil it.

Rules:
- Return JSON only. Do not wrap the JSON in markdown.
- Only use information explicitly stated in the text. Do not guess.
- For fields not mentioned, use null for scalars and [] for lists.
- Follow this JSON schema exactly:

{json.dumps(schema_dict, ensure_ascii=False, indent=2)}

Merged context:
{merged_context}
""".strip()


def rewrite_merged_context(
    merged_context: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[StandardizedJobQuery, dict]:
    """Call the LLM to rewrite merged_context into hard_filters + search_query.

    Returns (StandardizedJobQuery, usage_dict).
    Raises LLMParseError if the LLM output cannot be parsed/validated,
    or ValueError if the input is blank.
    """
    if not merged_context or not merged_context.strip():
        raise ValueError("merged_context is empty")

    prompt = _build_rewrite_prompt(merged_context)
    raw_output, usage = call_llm(prompt, provider=provider, model=model)
    parsed = extract_json(raw_output)

    try:
        query = StandardizedJobQuery.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(
            f"LLM output does not match StandardizedJobQuery schema: {exc}"
        ) from exc

    if not query.search_query.strip():
        raise LLMParseError("LLM returned an empty search_query")

    return query, usage
