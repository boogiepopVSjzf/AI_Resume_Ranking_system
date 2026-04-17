from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json


class ResumeScore(BaseModel):
    resume_id: str
    score: float = Field(ge=0, le=10)
    explanation: str


def build_resume_scoring_prompt(
    *,
    schema: dict[str, Any],
    feedback_examples: list[dict[str, Any]],
    resume: dict[str, Any],
) -> str:
    """Build the strict JSON scoring prompt for one resume."""
    examples_text = (
        json.dumps(feedback_examples, ensure_ascii=False, indent=2)
        if feedback_examples
        else "No reference feedback examples are available for this schema yet."
    )

    prompt_payload = {
        "scoring_schema": {
            "schema_id": schema.get("schema_id"),
            "schema_name": schema.get("schema_name"),
            "summary": schema.get("summary"),
            "rules_json": schema.get("rules_json", {}),
        },
        "target_resume": {
            "resume_id": resume.get("resume_id"),
            "metadata": resume.get("metadata", {}),
            "semantic_text": resume.get("semantic_text", ""),
        },
    }

    return f"""
You are a strict resume scoring assistant.

Score the target resume using only the provided scoring schema. The scoring schema rules are the source of truth.

You may also use reference feedback examples if available. These examples represent human preferences for the same schema.
If no examples are available, score only according to the schema rules.

Requirements:
1. Return JSON only. Do not wrap the JSON in markdown.
2. The JSON must contain exactly: resume_id, score, explanation.
3. score must be a number from 0 to 10.
4. explanation must be concise but specific.
5. explanation must explicitly reference the schema rules or criteria.
6. Do not use criteria outside the provided schema.
7. Do not invent resume facts.

Reference feedback examples:
{examples_text}

Scoring input:
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

Output JSON shape:
{{
  "resume_id": "{resume.get("resume_id", "")}",
  "score": 0.0,
  "explanation": "Explain the score by referencing the schema rules."
}}
""".strip()


def score_resume_with_schema(
    *,
    schema: dict[str, Any],
    feedback_examples: list[dict[str, Any]],
    resume: dict[str, Any],
) -> tuple[ResumeScore, dict]:
    """Score a single resume with the selected schema and optional examples."""
    prompt = build_resume_scoring_prompt(
        schema=schema,
        feedback_examples=feedback_examples,
        resume=resume,
    )
    raw_output, usage = call_llm(prompt)
    parsed = extract_json(raw_output)

    try:
        score = ResumeScore.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(f"LLM output does not match ResumeScore schema: {exc}") from exc

    if score.resume_id != resume.get("resume_id"):
        raise LLMParseError("LLM returned a resume_id that does not match the target resume")

    return score, usage
