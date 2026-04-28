from __future__ import annotations
from typing import Optional

from pydantic import ValidationError

from config import settings
from schemas.job_query import StandardizedJobQuery
from services.embedding_service import embed_text
from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json


SUPPORTED_FILTER_MODES = frozenset({"strict", "balanced", "semantic_only"})


def _build_rewrite_prompt_strict(merged_context: str) -> str:
    """Permissive prompt: any explicitly-mentioned requirement becomes a hard filter."""
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


def _build_rewrite_prompt_balanced(merged_context: str) -> str:
    """Conservative prompt: only mandatorily-required items become hard filters."""
    return f"""
You are a job-requirement analysis system.

Given the merged context below (which may contain an HR note and/or a job description),
produce a structured JSON object that decomposes the requirements into two parts:

1. **hard_filters** — STRICT, ELIMINATING criteria. A value goes here ONLY when the
   merged context EXPLICITLY marks that requirement as mandatory. Treat the following
   markers as "explicitly required":
   - English: "required", "must have", "must", "minimum", "at least", "no less than",
     "mandatory", "requirement(s):" sections, "hard requirement".
   - Chinese: "必须", "必备", "硬性要求", "强制", "要求", "至少", "不少于", "不低于",
     "需要"（仅当语义为强制要求，例如"需要至少3年..."）.

   If a requirement is only mentioned, recommended, preferred, or stated softly,
   DO NOT put it in hard_filters; it must go ONLY into search_query. Soft markers
   include (non-exhaustive):
   - English: "preferred", "nice to have", "ideally", "a plus", "bonus",
     "familiar with", "experience with", "exposure to", "knowledge of".
   - Chinese: "优先", "加分", "最好", "倾向", "了解", "熟悉", "有经验者优先".

   When in doubt about whether a requirement is mandatory, prefer leaving the
   hard_filter field empty (null / []) and capture the requirement in search_query.

   Fields:
   - min_yoe (integer | null): set ONLY when a minimum number of years of experience
     is explicitly required (e.g. "minimum 3 years", "at least 5 years", "至少3年",
     "不少于3年"). If years are only suggested or preferred, leave null.
   - required_skills (string[]): ONLY skills explicitly stated as required / mandatory.
     Normalise to concise lowercase tokens like "python", "sql", "fastapi"; deduplicate.
     Skills that are only "preferred", "familiar with", "experience with",
     "熟悉", "了解", "优先" must NOT appear here.
   - education_level (string | null): one of "high_school", "associate", "bachelor",
     "master", "phd", "other". Set ONLY when a minimum education level is explicitly
     required (e.g. "Bachelor's degree required", "本科及以上学历").
   - major (string | null): one of "computer_science", "mathematics", "medicine",
     "finance", "engineering", "other". Set ONLY when a specific major is explicitly
     required (e.g. "Computer Science major required", "要求计算机相关专业").

2. **search_query** — a concise natural-language sentence (one or two sentences)
   summarising the ideal candidate profile for semantic search. This MUST capture
   the full picture, INCLUDING soft / preferred / nice-to-have items that were
   intentionally excluded from hard_filters. Do NOT just copy the input; distil it.

Rules:
- Return JSON only. Do not wrap the JSON in markdown.
- Only use information explicitly stated in the text. Do not guess.
- For fields that are not mentioned OR not explicitly required, use null for
  scalars and [] for lists.
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


def filter_mode_to_prompt_variant(filter_mode: str) -> str:
    """Map a user-facing filter_mode to one of the two prompt variants.

    - "strict"        -> permissive prompt ("strict" variant)
    - "balanced"      -> conservative prompt ("balanced" variant)
    - "semantic_only" -> conservative prompt; the caller will discard hard_filters
                        before SQL anyway, so we share the cache with "balanced".
    """
    if filter_mode == "strict":
        return "strict"
    return "balanced"


def _build_rewrite_prompt(merged_context: str, *, filter_mode: str) -> str:
    if filter_mode_to_prompt_variant(filter_mode) == "strict":
        return _build_rewrite_prompt_strict(merged_context)
    return _build_rewrite_prompt_balanced(merged_context)


def rewrite_merged_context(
    merged_context: str,
    *,
    filter_mode: str = "balanced",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[StandardizedJobQuery, dict]:
    """Call the LLM to rewrite merged_context into hard_filters + search_query.

    The strictness of hard_filters is controlled by ``filter_mode`` at the prompt
    level: ``strict`` uses the original permissive prompt (any explicitly mentioned
    requirement becomes a hard filter), while ``balanced`` and ``semantic_only``
    use the conservative prompt (only items the JD/HR context explicitly marks
    as mandatory become hard filters).

    Returns (StandardizedJobQuery, usage_dict).
    Raises LLMParseError if the LLM output cannot be parsed/validated,
    or ValueError if the input is blank or filter_mode is unsupported.
    """
    if not merged_context or not merged_context.strip():
        raise ValueError("merged_context is empty")
    if filter_mode not in SUPPORTED_FILTER_MODES:
        raise ValueError(
            f"filter_mode must be one of: {', '.join(sorted(SUPPORTED_FILTER_MODES))}"
        )

    prompt = _build_rewrite_prompt(merged_context, filter_mode=filter_mode)
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
    return embed_text(query.search_query, for_query=True)
