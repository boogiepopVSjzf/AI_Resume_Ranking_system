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
    "excellent": (
        "Human believes this resume is a clearly strong match for the schema. "
        "It should rank near the top because it satisfies the most important rules "
        "with strong evidence and only limited weaknesses."
    ),
    "good": (
        "Human believes this resume is a solid match for the schema. "
        "It satisfies most important rules, but still has noticeable gaps, weaker evidence, "
        "or less depth than an excellent resume."
    ),
    "qualified": (
        "Human believes this resume is acceptable and could move forward, "
        "but it only meets the minimum bar or has meaningful weaknesses under multiple rules."
    ),
    "bad": (
        "Human believes this resume is a weak match for the schema. "
        "It misses important requirements, lacks convincing evidence under key rules, "
        "or should not rank highly against stronger candidates."
    ),
    "n/a": (
        "Human did not provide a scoring judgment. "
        "This label is audit-only and must never be used as a scoring calibration signal."
    ),
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


def _resume_input_from_bundle(
    bundle: dict[str, Any] | None,
    *,
    fallback_resume_id: str,
) -> dict[str, Any]:
    """Same shape as ``target_resume`` in the scoring prompt (for few-shot parity).

    Includes ``resume_id``, ``metadata``, and ``semantic_text`` — the compact
    representation used for LLM scoring. ``raw_json`` is intentionally omitted
    to keep prompts short.
    """
    if not bundle:
        return {
            "resume_id": fallback_resume_id,
            "metadata": {},
            "semantic_text": "",
        }
    rid = str(bundle.get("resume_id") or fallback_resume_id).strip() or fallback_resume_id
    meta = bundle.get("metadata")
    return {
        "resume_id": rid,
        "metadata": meta if isinstance(meta, dict) else {},
        "semantic_text": str(bundle.get("semantic_text") or "").strip(),
    }


def build_feedback_calibration_data(
    feedback_examples: list[dict[str, Any]],
    *,
    target_resume_id: str | None = None,
    resume_evidence_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a single few-shot list for the scoring LLM.

    Each element pairs (1) ``resume`` — same JSON shape as ``target_resume`` in
    the scoring prompt, when ``resume_evidence_by_id`` supplies DB rows — with
    (2) ``human_judgment`` — the saved feedback metadata for that example.

    When ``target_resume_id`` is set, ``human_judgment.same_resume_as_target``
    marks rows for the resume currently being scored.
    """
    few_shot_examples: list[dict[str, Any]] = []
    evidence = resume_evidence_by_id or {}

    for example in feedback_examples:
        if not isinstance(example, dict):
            continue

        label = str(example.get("label", "")).strip().lower()
        rid = str(example.get("resume_id") or "").strip()
        same_resume = bool(
            target_resume_id
            and rid == str(target_resume_id).strip()
        )
        bundle = evidence.get(rid) if rid else None
        few_shot_examples.append({
            "resume": _resume_input_from_bundle(bundle, fallback_resume_id=rid),
            "human_judgment": {
                "feedback_id": example.get("feedback_id"),
                "same_resume_as_target": same_resume,
                "human_label": label,
                "label_meaning": FEEDBACK_LABEL_MEANINGS.get(label, ""),
                "human_feedback": example.get("feedback_text") or "",
                "prior_model_score": example.get("score"),
                "prior_scoring_result_summary": _summarize_scoring_result(example.get("scoring_result")),
                "created_at": example.get("created_at"),
            },
        })

    same_resume_rows = [
        row for row in few_shot_examples
        if (row.get("human_judgment") or {}).get("same_resume_as_target")
    ]
    other_rows = [
        row for row in few_shot_examples
        if not (row.get("human_judgment") or {}).get("same_resume_as_target")
    ]
    ts_key = lambda row: str((row.get("human_judgment") or {}).get("created_at") or "")
    same_resume_rows.sort(key=ts_key, reverse=True)
    other_rows.sort(key=ts_key, reverse=True)
    ordered = same_resume_rows + other_rows

    return {
        "label_meanings": FEEDBACK_LABEL_MEANINGS,
        "few_shot_examples": ordered,
    }


def _has_usable_feedback_calibration(feedback_calibration_data: dict[str, Any]) -> bool:
    for row in feedback_calibration_data.get("few_shot_examples") or []:
        hj = row.get("human_judgment") or {}
        label = str(hj.get("human_label", "")).strip().lower()
        if label in {"excellent", "good", "qualified", "bad"}:
            return True
    return False


def _same_resume_human_tier_instructions(feedback_calibration_data: dict[str, Any]) -> str:
    """When this resume has a saved human tier, require weighted total to align with it.

    ``few_shot_examples`` is ordered with same-resume rows first (newest first).
    """
    label: str | None = None
    for row in feedback_calibration_data.get("few_shot_examples") or []:
        hj = row.get("human_judgment") or {}
        if not hj.get("same_resume_as_target"):
            continue
        raw = str(hj.get("human_label", "")).strip().lower()
        if raw in {"excellent", "good", "qualified", "bad"}:
            label = raw
            break
    if not label:
        return ""

    bands = {
        "excellent": "7.5–10.0",
        "good": "6.0–8.5",
        "qualified": "4.0–6.5",
        "bad": "0.0–4.5",
    }
    band = bands[label]
    return f"""
**Human tier alignment (this resume only):** At least one few-shot row has
`human_judgment.same_resume_as_target: true` with `human_label` = `{label}`.
That is a **binding hiring prior** for the same candidate you are scoring now.

Set per-rule scores so that, once each rule score is multiplied by its schema
weight and summed (the same weighted total the backend will compute), the
**implied weighted total** falls **approximately in {band}** on the 0–10 scale.

You may only land **outside** that band if concrete resume evidence under
**specific rules** makes the tier indefensible — then lower or raise the
affected rules, cite those rules in `reason` / `feedback_influence`, and explain
the conflict clearly in `explanation`. Do **not** reproduce an old automated
total that plainly conflicts with `{label}` without such rule-level
justification.
""".strip()


def _feedback_mode_instructions(mode: str) -> str:
    if mode == "off":
        return ""
    return """
The calibration JSON contains `few_shot_examples`. Each item has:
- `resume` — same keys as `Scoring input.target_resume` (`resume_id`, `metadata`, `semantic_text`).
- `human_judgment` — label, free-text note, prior model score snapshot, etc.

Use few-shots where `same_resume_as_target` is false as **rule-calibration**
guidance only — do not copy those rows' human labels or implied totals onto this
candidate. When a **Human tier alignment** block appears below for this resume,
it overrides generic "do not copy label" guidance for **overall score level**.

Rules:
- The scoring schema and the target resume evidence remain the source of truth.
- Read each few-shot's `resume` plus `human_judgment` to calibrate how strict or
  lenient each rule should be when evidence is strong, borderline, or weak.
- Rows with `human_judgment.same_resume_as_target: true` identify **this same
  resume**; when the **Human tier alignment** block is present, follow it for
  weighted-total expectations; otherwise they are the strongest qualitative prior.
- Rows with `human_judgment.human_label` of `n/a` are audit-only: never use them
  as a scoring signal.
- Do not mimic another row's label just because of surface similarity.
- If a few-shot row changes your strictness on a rule, set `feedback_used=true`
  for that rule and cite `human_judgment.feedback_id` in `feedback_influence`.

`feedback_usage_summary`: list which `feedback_id`s materially informed any rule
in `used_feedback_ids`; put unused ids (including `n/a` rows) in
`ignored_feedback_ids`; one-sentence `overall_influence`.

If `few_shot_examples` is empty or every row is `n/a`, set every rule
`feedback_used=false` and note that no usable few-shots were present.
""".strip()


def build_resume_scoring_prompt(
    *,
    schema: dict[str, Any],
    feedback_examples: list[dict[str, Any]],
    resume: dict[str, Any],
    feedback_influence_mode: str = "on",
    resume_evidence_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build the strict JSON scoring prompt for one resume."""
    feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    feedback_calibration_data = build_feedback_calibration_data(
        feedback_examples,
        target_resume_id=resume.get("resume_id"),
        resume_evidence_by_id=resume_evidence_by_id,
    )
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

    meta = resume.get("metadata")
    prompt_payload = {
        "scoring_schema": {
            "schema_id": schema.get("schema_id"),
            "schema_name": schema.get("schema_name"),
            "summary": schema.get("summary"),
            "rules_json": rules_json,
        },
        "target_resume": {
            "resume_id": resume.get("resume_id"),
            "metadata": meta if isinstance(meta, dict) else {},
            "semantic_text": str(resume.get("semantic_text") or "").strip(),
        },
    }

    feedback_instructions = _feedback_mode_instructions(feedback_influence_mode)
    tier_extra = ""
    if feedback_instructions:
        tier_extra = _same_resume_human_tier_instructions(feedback_calibration_data)
    feedback_section = (
        f"""
Few-shot calibration guidance:
{feedback_instructions}
{tier_extra}

Few-shot calibration data:
{examples_text}
"""
        if feedback_instructions
        else ""
    )

    return f"""
You are a strict resume scoring assistant.

Score the target resume using only the provided scoring schema. The scoring schema rules are the source of truth.

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
{feedback_section}
Scoring input:
{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}

Output JSON shape:
{{
  "resume_id": "{resume.get("resume_id", "")}",
  "rule_scores": {json.dumps(rule_score_shape, ensure_ascii=False, indent=2)},
  "feedback_usage_summary": {{
    "used_feedback_ids": [],
    "ignored_feedback_ids": [],
    "overall_influence": "Summarize how few-shot calibration affected this score, or state none was used."
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
    resume_evidence_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[ResumeScore, dict]:
    """Score a single resume with the selected schema and optional examples."""
    feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    prompt = build_resume_scoring_prompt(
        schema=schema,
        feedback_examples=feedback_examples,
        resume=resume,
        feedback_influence_mode=feedback_influence_mode,
        resume_evidence_by_id=resume_evidence_by_id,
    )
    raw_output, usage = call_llm(
        prompt,
        provider=settings.SCORING_LLM_PROVIDER,
        model=settings.SCORING_LLM_MODEL,
    )
    parsed = extract_json(raw_output)
    parsed = _calculate_weighted_score(parsed=parsed, schema=schema)
    if feedback_influence_mode == "off":
        for rule_score in parsed["rule_scores"].values():
            rule_score["feedback_used"] = False
            rule_score["feedback_influence"] = ""
        parsed["feedback_usage_summary"] = {
            "used_feedback_ids": [],
            "ignored_feedback_ids": [],
            "overall_influence": "Few-shot calibration was disabled.",
        }
    elif not _has_usable_feedback_calibration(
        build_feedback_calibration_data(
            feedback_examples,
            target_resume_id=resume.get("resume_id"),
            resume_evidence_by_id=resume_evidence_by_id,
        )
    ):
        for rule_score in parsed["rule_scores"].values():
            rule_score["feedback_used"] = False
            rule_score["feedback_influence"] = ""
        parsed["feedback_usage_summary"] = {
            "used_feedback_ids": [],
            "ignored_feedback_ids": [],
            "overall_influence": "No usable few-shot examples were available.",
        }

    try:
        score = ResumeScore.model_validate(parsed)
    except ValidationError as exc:
        raise LLMParseError(f"LLM output does not match ResumeScore schema: {exc}") from exc

    if score.resume_id != resume.get("resume_id"):
        raise LLMParseError("LLM returned a resume_id that does not match the target resume")

    return score, usage
