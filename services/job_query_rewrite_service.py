from __future__ import annotations
from typing import Optional

from pydantic import ValidationError

from config import settings
from schemas.job_query import StandardizedJobQuery
from services.embedding_service import embed_text
from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json


def _build_rewrite_prompt(merged_context: str) -> str:
    return f"""
You are a job-requirement analysis system.

Given the merged context below (which may contain an HR note and/or a job description),
produce a structured JSON object that decomposes the requirements into two parts:

1. **hard_filters** — deterministic filtering criteria:
   - min_yoe (integer or null): minimum years of experience required.
   - required_skills (list of strings): key skills explicitly mentioned; normalise to concise lowercase skill tokens like "python", "sql", "fastapi"; deduplicate.
   - education_level (string or null): one of "high_school", "associate", "bachelor", "master", "phd", "other".
   - major (string or null): one of "computer_science", "mathematics", "medicine", "finance", "engineering", "other".
2. **search_query** — a concise natural-language sentence (one or two sentences) summarising the ideal candidate profile for semantic search. Do NOT just copy the input; distil it.

Rules:
- Return JSON only. Do not wrap the JSON in markdown.
- Only use information explicitly stated in the text. Do not guess.
- For fields not mentioned, use null for scalars and [] for lists.
- Follow this exact JSON shape:
{{
  "hard_filters": {{
    "min_yoe": integer | null,
    "required_skills": string[],
    "education_level": "high_school" | "associate" | "bachelor" | "master" | "phd" | "other" | null,
    "major": "computer_science" | "mathematics" | "medicine" | "finance" | "engineering" | "other" | null
  }},
  "search_query": string
}}

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
    raw_output, usage = call_llm(
        prompt,
        provider=provider or settings.PARSE_QUERY_LLM_PROVIDER,
        model=model or settings.PARSE_QUERY_LLM_MODEL,
    )
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


def embed_search_query(query: StandardizedJobQuery) -> Optional[list[float]]:
    """Embed the semantic search_query text for vector retrieval."""
    return embed_text(query.search_query)
