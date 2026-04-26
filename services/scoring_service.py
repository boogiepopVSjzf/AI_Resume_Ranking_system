from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from config import settings
from services.llm_service import call_llm
from utils.errors import LLMParseError
from utils.llm_json import extract_json

FEEDBACK_INFLUENCE_MODES = {"off", "on"}

FEEDBACK_LABEL_MEANINGS = {
    "excellent": "Human believes this resume strongly satisfies the schema and should score near the top.",
    "good": "Human believes this resume satisfies most schema requirements with minor gaps.",
    "qualified": "Human believes this resume is acceptable but has meaningful gaps.",
    "bad": "Human believes this resume should receive a low score for this schema.",
    "n/a": "Human did not provide a scoring judgment. This is stored for audit only and must not calibrate scoring.",
}


class RuleScore(BaseModel):
    score: float = Field(ge=0, le=10)
    reason: str
    weight: float = Field(default=0, ge=0, le=1)
    weighted_score: float = Field(default=0, ge=0, le=10)
    feedback_used: bool = False
    feedback_influence: str = ""
    # Removed feedback_used and feedback_influence display logic from the frontend.


class FeedbackUsageSummary(BaseModel):
    mode: str = "on"
    used_feedback_ids: list[str] = Field(default_factory=list)
    ignored_feedback_ids: list[str] = Field(default_factory=list)
    overall_influence: str = ""


class ResumeScore(BaseModel):
    resume_id: str
    score: float = Field(ge=0, le=10)
    rule_scores: dict[str, RuleScore]
    feedback_usage_summary: FeedbackUsageSummary = Field(default_factory=FeedbackUsageSummary)
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
            "feedback_used": bool(raw_rule_score.get("feedback_used", False)),
            "feedback_influence": str(raw_rule_score.get("feedback_influence", "")).strip(),
        }

    if missing_rules:
        raise LLMParseError(f"LLM output is missing rule_scores for: {', '.join(missing_rules)}")

    parsed["rule_scores"] = normalized_rule_scores
    parsed["score"] = round(final_score, 4)
    if not isinstance(parsed.get("feedback_usage_summary"), dict):
        parsed["feedback_usage_summary"] = {}
    return parsed


def normalize_feedback_influence_mode(mode: str | None) -> str:
    """Return a supported feedback influence mode."""
    normalized = str(mode or "on").strip().lower()
    if normalized not in FEEDBACK_INFLUENCE_MODES:
        raise ValueError("feedback_influence_mode must be one of: off, on")
    return normalized


def _summarize_scoring_result(scoring_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scoring_result, dict):
        return {}

    rule_scores = scoring_result.get("rule_scores")
    compact_rules = {}
    if isinstance(rule_scores, dict):
        for rule_key, rule_score in rule_scores.items():
            if not isinstance(rule_score, dict):
                continue
            compact_rules[rule_key] = {
                "score": rule_score.get("score"),
                "reason": rule_score.get("reason"),
            }

    return {
        "total_score": scoring_result.get("score"),
        "rule_scores": compact_rules,
        "explanation": scoring_result.get("explanation"),
    }


def build_feedback_calibration_data(
    feedback_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert raw saved feedback rows into structured calibration data for prompting."""
    calibration_data: dict[str, Any] = {
        "label_meanings": FEEDBACK_LABEL_MEANINGS,
        "positive_examples": [],
        "negative_examples": [],
        "neutral_unrated_examples": [],
    }

    for example in feedback_examples:
        if not isinstance(example, dict):
            continue

        label = str(example.get("label", "")).strip().lower()
        structured_example = {
            "feedback_id": example.get("feedback_id"),
            "resume_id": example.get("resume_id"),
            "human_label": label,
            "label_meaning": FEEDBACK_LABEL_MEANINGS.get(label, ""),
            "human_feedback": example.get("feedback_text") or "",
            "prior_model_score": example.get("score"),
            "prior_scoring_result_summary": _summarize_scoring_result(example.get("scoring_result")),
            "created_at": example.get("created_at"),
        }

        if label in {"excellent", "good", "qualified"}:
            calibration_data["positive_examples"].append(structured_example)
        elif label == "bad":
            calibration_data["negative_examples"].append(structured_example)
        elif label == "n/a":
            calibration_data["neutral_unrated_examples"].append(structured_example)

    return calibration_data


def _has_usable_feedback_calibration(feedback_calibration_data: dict[str, Any]) -> bool:
    return bool(
        feedback_calibration_data.get("positive_examples")
        or feedback_calibration_data.get("negative_examples")
    )


def _feedback_mode_instructions(mode: str) -> str:
    if mode == "off":
        return """
Feedback influence mode: off.
Ignore feedback calibration data completely. Score only according to the scoring schema.
Return feedback_used=false for every rule and an empty feedback_usage_summary.
""".strip()
    return """
Feedback influence mode: on.
You must inspect and use all usable feedback calibration examples as human preference calibration.
The scoring schema remains the source of truth.
Compare the target resume against feedback examples before finalizing each rule score.
Use feedback when examples involve similar evidence, missing evidence, strengths, or weaknesses under the same schema.
If a relevant human label conflicts with a prior model score, prefer the human label as the calibration signal.
Do not copy scores from examples. Adjust only when the target resume has comparable evidence under the same rule.
Ignore n/a examples as scoring signals because they are audit-only.
For each rule, state whether feedback changed the score, confirmed the score, or was irrelevant.
The feedback_usage_summary must name used_feedback_ids, ignored_feedback_ids, and the overall influence.
If no usable feedback calibration examples are detected, set feedback_used=false for every rule and explain that no feedback was detected.
""".strip()


def build_resume_scoring_prompt(
    *,
    schema: dict[str, Any],
    feedback_examples: list[dict[str, Any]],
    resume: dict[str, Any],
    feedback_influence_mode: str = "on",
) -> str:
    """Build the strict JSON scoring prompt for one resume."""
    feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    feedback_calibration_data = build_feedback_calibration_data(feedback_examples)
    has_usable_feedback = _has_usable_feedback_calibration(feedback_calibration_data)
    examples_text = (
        json.dumps(feedback_calibration_data, ensure_ascii=False, indent=2)
        if has_usable_feedback and feedback_influence_mode == "on"
        else (
            "Feedback influence mode is on, but no usable feedback calibration examples "
            "were detected. n/a feedback entries, if present, are audit-only and must not "
            "calibrate scoring."
            if feedback_influence_mode == "on"
            else "No feedback calibration data should be used for this scoring run."
        )
    )

    rules_json = _rules_for_prompt(schema)
    rule_score_shape = {
        rule_key: {
            "score": 0.0,
            "reason": f"Explain the score for {rule_key} using only this rule.",
            "feedback_used": False,
            "feedback_influence": "Describe whether structured feedback calibration influenced this rule.",
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

Feedback calibration:
{_feedback_mode_instructions(feedback_influence_mode)}

Requirements:
1. Return JSON only. Do not wrap the JSON in markdown.
2. The JSON must contain exactly: resume_id, rule_scores, feedback_usage_summary, explanation.
3. rule_scores must contain one object for every rule key in rules_json, using the same keys such as rules1, rules2.
4. Each rule score must be a number from 0 to 10.
5. Each rule reason must be concise, specific, and based only on evidence in the resume.
6. Each rule must include feedback_used and feedback_influence.
7. Do not use criteria outside the provided schema.
8. Do not invent resume facts.
9. Do not calculate the final weighted total score. The backend will calculate it from rule weights.
10. explanation must summarize the overall fit and explicitly reference the schema rules or criteria.

Structured feedback calibration data:
{examples_text}

Scoring input:
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

Output JSON shape:
{{
  "resume_id": "{resume.get("resume_id", "")}",
  "rule_scores": {json.dumps(rule_score_shape, ensure_ascii=False, indent=2)},
  "feedback_usage_summary": {{
    "mode": "{feedback_influence_mode}",
    "used_feedback_ids": [],
    "ignored_feedback_ids": [],
    "overall_influence": "Summarize how feedback calibration affected or did not affect this score."
  }},
  "explanation": "Explain the score by referencing the schema rules."
}}
""".strip()


def score_resume_with_schema(
    *,
    schema: dict[str, Any],
    feedback_examples: list[dict[str, Any]],
    resume: dict[str, Any],
    feedback_influence_mode: str = "on",
) -> tuple[ResumeScore, dict]:
    """Score a single resume with the selected schema and optional examples."""
    feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    prompt = build_resume_scoring_prompt(
        schema=schema,
        feedback_examples=feedback_examples,
        resume=resume,
        feedback_influence_mode=feedback_influence_mode,
    )
    raw_output, usage = call_llm(
        prompt,
        provider=settings.SCORING_LLM_PROVIDER,
        model=settings.SCORING_LLM_MODEL,
    )
    parsed = extract_json(raw_output)
    parsed = _calculate_weighted_score(parsed=parsed, schema=schema)
    parsed["feedback_usage_summary"] = {
        **parsed.get("feedback_usage_summary", {}),
        "mode": feedback_influence_mode,
    }
    if feedback_influence_mode == "off":
        for rule_score in parsed["rule_scores"].values():
            rule_score["feedback_used"] = False
            rule_score["feedback_influence"] = ""
        parsed["feedback_usage_summary"] = {
            "mode": "off",
            "used_feedback_ids": [],
            "ignored_feedback_ids": [],
            "overall_influence": "Feedback influence mode was off, so no feedback calibration was used.",
        }
    elif not _has_usable_feedback_calibration(build_feedback_calibration_data(feedback_examples)):
        for rule_score in parsed["rule_scores"].values():
            rule_score["feedback_used"] = False
            rule_score["feedback_influence"] = "No usable feedback calibration examples were detected."
        parsed["feedback_usage_summary"] = {
            "mode": "on",
            "used_feedback_ids": [],
            "ignored_feedback_ids": [],
            "overall_influence": "Feedback influence mode was on, but no usable feedback calibration examples were detected.",
        }

    try:
        score = ResumeScore.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(f"LLM output does not match ResumeScore schema: {exc}") from exc

    if score.resume_id != resume.get("resume_id"):
        raise LLMParseError("LLM returned a resume_id that does not match the target resume")

    return score, usage
