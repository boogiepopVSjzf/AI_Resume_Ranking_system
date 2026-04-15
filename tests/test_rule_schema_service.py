from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from services.rule_schema_service import build_rule_schema_result, describe_rule
from utils.errors import LLMParseError


RULE = {
    "min_yoe": 3,
    "required_skills": ["python", "fastapi"],
    "education_level": "bachelor",
}


class TestDescribeRule:
    @patch("services.rule_schema_service.call_llm")
    def test_returns_llm_description(self, mock_llm):
        mock_llm.return_value = (
            json.dumps({
                "rule_description": "Data science candidates need 3+ years of experience, Python, FastAPI, and a bachelor's degree."
            }),
            {"total_tokens": 80},
        )

        description, usage = describe_rule(RULE, "data science")

        assert "3+ years" in description
        assert usage == {"total_tokens": 80}

    @patch("services.rule_schema_service.call_llm")
    def test_invalid_llm_output_raises_parse_error(self, mock_llm):
        mock_llm.return_value = (json.dumps({"wrong": "shape"}), {})

        with pytest.raises(LLMParseError, match="RuleDescriptionLLMOutput"):
            describe_rule(RULE, "data science")


class TestBuildRuleSchemaResult:
    @patch("services.rule_schema_service.uuid4")
    @patch("services.rule_schema_service.embed_text", return_value=[0.1, 0.2, 0.3])
    @patch("services.rule_schema_service.describe_rule")
    def test_builds_table_shaped_response(self, mock_describe, mock_embed, mock_uuid):
        mock_describe.return_value = ("Must have Python and FastAPI experience.", {})
        mock_uuid.return_value.hex = "schema123"

        result, usage = build_rule_schema_result(RULE, "software engineering")

        assert result.schema_id == "schema123"
        assert result.rule_json == RULE
        assert result.rule_description == "Must have Python and FastAPI experience."
        assert result.embedding_vector == [0.1, 0.2, 0.3]
        assert result.job_name == "software engineering"
        assert usage == {}
        mock_embed.assert_called_once_with(
            "Job: software engineering\nRule: Must have Python and FastAPI experience."
        )

    def test_empty_rule_rejected(self):
        with pytest.raises(ValueError, match="rule"):
            build_rule_schema_result({}, "software engineering")

    def test_empty_job_name_rejected(self):
        with pytest.raises(ValueError, match="job_name"):
            build_rule_schema_result(RULE, "   ")
