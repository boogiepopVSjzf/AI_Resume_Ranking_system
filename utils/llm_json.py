from __future__ import annotations

import json
import re

from utils.errors import LLMParseError


def extract_json(raw_output: str) -> dict:
    """Extract a JSON object from raw LLM output."""
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
