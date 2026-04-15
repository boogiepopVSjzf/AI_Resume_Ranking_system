from __future__ import annotations

import json
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import ValidationError

from schemas.final_result import RuleDescriptionLLMOutput, RuleSchemaResponse
from services.embedding_service import embed_text
from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json


def _normalise_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Round-trip through JSON so the response only contains JSON-safe values."""
    return json.loads(json.dumps(rule, ensure_ascii=False))


def _rule_to_text(rule: Dict[str, Any]) -> str:
    return json.dumps(rule, ensure_ascii=False, sort_keys=True, indent=2)


def _build_rule_description_prompt(rule: Dict[str, Any], job_name: str) -> str:
    rule_text = _rule_to_text(rule)
    return f"""
You are converting machine-readable recruiting rules into a concise natural-language rule.

Job name:
{job_name}

JSON rule:
{rule_text}

Return JSON only. Do not wrap it in markdown.
Follow this exact JSON shape:
{{
  "rule_description": "one concise natural-language description of the rule"
}}

Rules:
- Preserve every explicit condition in the JSON rule.
- Do not invent requirements that are not present.
- Mention the job name only if it helps the sentence read naturally.
- Prefer clear HR-facing language over technical JSON wording.
""".strip()


def describe_rule(
    rule: Dict[str, Any],
    job_name: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, dict]:
    prompt = _build_rule_description_prompt(rule, job_name)
    raw_output, usage = call_llm(prompt, provider=provider, model=model)
    parsed = extract_json(raw_output)

    try:
        output = RuleDescriptionLLMOutput.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(
            f"LLM output does not match RuleDescriptionLLMOutput schema: {exc}"
        ) from exc

    description = output.rule_description.strip()
    if not description:
        raise LLMParseError("LLM returned an empty rule_description")

    return description, usage


def build_rule_schema_result(rule: Dict[str, Any], job_name: str) -> tuple[RuleSchemaResponse, dict]:
    if not job_name or not job_name.strip():
        raise ValueError("job_name is required and must be a non-empty string")
    if not isinstance(rule, dict) or not rule:
        raise ValueError("rule is required and must be a non-empty JSON object")

    normalised_rule = _normalise_rule(rule)
    clean_job_name = job_name.strip()
    rule_description, usage = describe_rule(normalised_rule, clean_job_name)
    embedding_input = f"Job: {clean_job_name}\nRule: {rule_description}"
    embedding = embed_text(embedding_input)
    if not embedding:
        raise ValueError("embedding_vector is empty")

    return (
        RuleSchemaResponse(
            schema_id=uuid4().hex,
            rule_json=normalised_rule,
            rule_description=rule_description,
            embedding_vector=embedding,
            job_name=clean_job_name,
        ),
        usage,
    )
