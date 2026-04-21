from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from config import settings
from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json


class RuleScore(BaseModel):
    score: float = Field(ge=0, le=10)
    reason: str
    weight: float = Field(default=0, ge=0, le=1)
    weighted_score: float = Field(default=0, ge=0, le=10)


class ResumeScore(BaseModel):
    resume_id: str
    score: float = Field(ge=0, le=10)
    rule_scores: dict[str, RuleScore]
    explanation: str


def _rules_for_prompt(schema: dict[str, Any]) -> dict[str, Any]:
    rules_json = schema.get("rules_json", {})
    if isinstance(rules_json, dict):
        return rules_json
    return {}


def _calculate_weighted_score(
    *,
    parsed: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Trust the LLM for per-rule judgments, but calculate the total in code."""
    rules_json = _rules_for_prompt(schema)
    raw_rule_scores = parsed.get("rule_scores")
    if not isinstance(raw_rule_scores, dict):
        raise LLMParseError("LLM output must include rule_scores as an object")

    normalized_rule_scores: dict[str, dict[str, Any]] = {}
    final_score = 0.0
    missing_rules = []

    for rule_key, rule in rules_json.items():
        if rule_key not in raw_rule_scores:
            missing_rules.append(rule_key)
            continue

        raw_rule_score = raw_rule_scores[rule_key]
        if not isinstance(raw_rule_score, dict):
            raise LLMParseError(f"rule_scores.{rule_key} must be an object")

        try:
            rule_score = float(raw_rule_score.get("score"))
        except (TypeError, ValueError) as exc:
            raise LLMParseError(f"rule_scores.{rule_key}.score must be a number") from exc
        if rule_score < 0 or rule_score > 10:
            raise LLMParseError(f"rule_scores.{rule_key}.score must be between 0 and 10")

        weight = float(rule.get("weight", 0) or 0)
        weighted_score = round(rule_score * weight, 4)
        final_score += weighted_score
        normalized_rule_scores[rule_key] = {
            "score": round(rule_score, 4),
            "weight": weight,
            "weighted_score": weighted_score,
            "reason": str(raw_rule_score.get("reason", "")).strip(),
        }

    if missing_rules:
        raise LLMParseError(f"LLM output is missing rule_scores for: {', '.join(missing_rules)}")

    parsed["rule_scores"] = normalized_rule_scores
    parsed["score"] = round(final_score, 4)
    return parsed


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

    rules_json = _rules_for_prompt(schema)
    rule_score_shape = {
        rule_key: {
            "score": 0.0,
            "reason": f"Explain the score for {rule_key} using only this rule.",
        }
        for rule_key in rules_json
    }

    prompt_payload = {
        "scoring_schema": {
            "schema_id": schema.get("schema_id"),
            "schema_name": schema.get("schema_name"),
            "summary": schema.get("summary"),
            "rules_json": rules_json,
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
2. The JSON must contain exactly: resume_id, rule_scores, explanation.
3. rule_scores must contain one object for every rule key in rules_json, using the same keys such as rules1, rules2.
4. Each rule score must be a number from 0 to 10.
5. Each rule reason must be concise, specific, and based only on evidence in the resume.
6. explanation must summarize the overall fit and explicitly reference the schema rules or criteria.
7. Do not use criteria outside the provided schema.
8. Do not invent resume facts.
9. Do not calculate the final weighted total score. The backend will calculate it from rule weights.

Reference feedback examples:
{examples_text}

Scoring input:
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

Output JSON shape:
{{
  "resume_id": "{resume.get("resume_id", "")}",
  "rule_scores": {json.dumps(rule_score_shape, ensure_ascii=False, indent=2)},
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
    raw_output, usage = call_llm(
        prompt,
        provider=settings.SCORING_LLM_PROVIDER,
        model=settings.SCORING_LLM_MODEL,
    )
    parsed = extract_json(raw_output)
    parsed = _calculate_weighted_score(parsed=parsed, schema=schema)

    try:
        score = ResumeScore.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(f"LLM output does not match ResumeScore schema: {exc}") from exc

    if score.resume_id != resume.get("resume_id"):
        raise LLMParseError("LLM returned a resume_id that does not match the target resume")

    return score, usage
