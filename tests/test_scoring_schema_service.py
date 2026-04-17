from __future__ import annotations

import pytest

from services.scoring_schema_service import parse_rules_text


def test_parse_rules_text_uses_rule_keys_without_rule_id():
    rules_json = parse_rules_text(
        "rules1: ML fundamentals weight: 25% "
        "rules2: Python programming weight: 20% "
        "rules3: MLOps awareness weight: 0.1"
    )

    assert set(rules_json) == {"rules1", "rules2", "rules3"}
    assert rules_json["rules1"] == {
        "rule_text": "ML fundamentals",
        "weight": 0.25,
        "raw_text": "ML fundamentals weight: 25%",
    }
    assert "rule_id" not in rules_json["rules1"]


def test_parse_rules_text_requires_weight_for_each_rule():
    with pytest.raises(ValueError, match="rules2 must include a weight"):
        parse_rules_text("rules1: ML fundamentals weight: 25% rules2: Python programming")
