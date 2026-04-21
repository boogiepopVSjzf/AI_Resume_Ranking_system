from __future__ import annotations

import json
import re
from typing import Any, Optional

from config import settings
from services.embedding_service import embed_text
from services.llm_service import call_llm
from utils.errors import LLMParseError


_RULE_HEADER_RE = re.compile(
    r"(?is)\b(rules\d+)\s*[:：]\s*(.*?)(?=\brules\d+\s*[:：]|\Z)",
    re.DOTALL,
)
_WEIGHT_RE = re.compile(
    r"(?i)\bweight\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?|\(\s*(\d+(?:\.\d+)?)\s*%\s*\)"
)
_WEIGHT_SUM_TOLERANCE = 0.0001


def parse_rules_text(rules_text: str) -> dict[str, Any]:
    """Parse user-provided rules1/rules2/... text into structured rules_json."""
    if not rules_text or not rules_text.strip():
        raise ValueError("rules cannot be empty")

    rules_json: dict[str, Any] = {}
    for match in _RULE_HEADER_RE.finditer(rules_text.strip()):
        rule_key = match.group(1).lower()
        raw_rule_text = match.group(2).strip()
        if not raw_rule_text:
            continue
        if rule_key in rules_json:
            raise ValueError(f"duplicate rule key: {rule_key}")
        weight = _extract_weight(raw_rule_text)
        if weight is None:
            raise ValueError(
                f"{rule_key} must include a weight, for example `weight: 25%` or `weight: 0.25`"
            )
        rule_text = _strip_weight_marker(raw_rule_text).strip(" ,;.-")
        rules_json[rule_key] = {
            "rule_text": rule_text or raw_rule_text,
            "weight": weight,
            "raw_text": raw_rule_text,
        }

    if not rules_json:
        raise ValueError("rules must be provided as rules1:, rules2:, rules3:, ...")

    weight_sum = sum(rule["weight"] for rule in rules_json.values())
    if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"rule weights must add up to 100%; current total is {round(weight_sum * 100, 4)}%"
        )

    return rules_json


def _extract_weight(text: str) -> Optional[float]:
    match = _WEIGHT_RE.search(text)
    if not match:
        return None
    value = float(match.group(1) or match.group(2))
    if match.group(1) and value <= 1 and "%" not in match.group(0):
        return round(value, 6)
    return round(value / 100, 6)


def _strip_weight_marker(text: str) -> str:
    return _WEIGHT_RE.sub("", text)


def build_schema_summary_prompt(schema_name: str, rules_json: dict[str, Any]) -> str:
    return f"""
You are a scoring schema summarization assistant for an AI resume ranking system.

Your task is to summarize a set of job-specific scoring rules into a concise schema summary.

The input contains:
- schema_name: the name of the scoring schema
- rules_json: a structured JSON object containing multiple scoring rules

rules_json is keyed by rules1, rules2, rules3, etc.
Each rule includes:
- rule_text
- weight
- raw_text, preserving the user's original input

Instructions:
1. Summarize only the information explicitly present in the rules.
2. Do not invent new evaluation criteria.
3. Do not change the meaning of any rule.
4. Preserve the relative importance of rules if weights are provided.
5. The summary should describe what this scoring schema evaluates and how candidates should be judged.
6. Use clear, professional English.
7. Keep the summary concise, ideally 4-8 sentences.
8. Do not output markdown.
9. Do not output JSON.
10. Output only the plain-text summary.

schema_name:
{schema_name}

rules_json:
{json.dumps(rules_json, ensure_ascii=False, indent=2)}

Summary:
""".strip()


def summarize_rules(schema_name: str, rules_json: dict[str, Any]) -> tuple[str, dict]:
    prompt = build_schema_summary_prompt(schema_name, rules_json)
    summary, usage = call_llm(
        prompt,
        provider=settings.SCHEMA_LLM_PROVIDER,
        model=settings.SCHEMA_LLM_MODEL,
    )
    summary = summary.strip()
    if not summary:
        raise LLMParseError("LLM returned an empty scoring schema summary")
    return summary, usage


def build_scoring_schema_payload(schema_name: str, rules_text: str) -> tuple[dict[str, Any], dict]:
    normalized_name = schema_name.strip()
    if not normalized_name:
        raise ValueError("schema_name cannot be empty")

    rules_json = parse_rules_text(rules_text)
    summary, usage = summarize_rules(normalized_name, rules_json)
    embedding = embed_text(summary)
    return {
        "schema_name": normalized_name,
        "rules_json": rules_json,
        "summary": summary,
        "embedding": embedding,
    }, usage
