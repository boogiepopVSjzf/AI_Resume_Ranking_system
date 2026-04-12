from __future__ import annotations

import json

import pytest
from unittest.mock import patch

from schemas.job_query import HardFilters, StandardizedJobQuery
from services.job_query_rewrite_service import embed_search_query, rewrite_merged_context
from utils.errors import LLMParseError


VALID_LLM_OUTPUT = json.dumps({
    "hard_filters": {
        "min_yoe": 3,
        "required_skills": ["python", "fastapi", "postgresql"],
        "education_level": "bachelor",
        "major": "computer_science",
    },
    "search_query": "有3年以上Python后端经验的工程师，熟悉FastAPI和PostgreSQL，本科及以上学历，坐标上海。",
})

MINIMAL_LLM_OUTPUT = json.dumps({
    "hard_filters": {
        "min_yoe": None,
        "required_skills": [],
        "education_level": None,
        "major": None,
    },
    "search_query": "General software engineer.",
})

MERGED_CONTEXT = (
    "HR_NOTE:\n请重点看项目经验\n\n"
    "JOB_DESCRIPTION:\n需要3年以上Python后端经验，熟悉FastAPI和PostgreSQL，本科及以上，上海。"
)


class TestRewriteMergedContext:
    @patch("services.job_query_rewrite_service.call_llm")
    def test_returns_standardized_query(self, mock_llm):
        mock_llm.return_value = (VALID_LLM_OUTPUT, {"total_tokens": 150})
        query, usage = rewrite_merged_context(MERGED_CONTEXT)

        assert isinstance(query, StandardizedJobQuery)
        assert isinstance(query.hard_filters, HardFilters)
        assert usage == {"total_tokens": 150}

    @patch("services.job_query_rewrite_service.call_llm")
    def test_hard_filters_fields(self, mock_llm):
        mock_llm.return_value = (VALID_LLM_OUTPUT, {})
        query, _ = rewrite_merged_context(MERGED_CONTEXT)

        assert query.hard_filters.min_yoe == 3
        assert "python" in query.hard_filters.required_skills
        assert query.hard_filters.education_level == "bachelor"
        assert query.hard_filters.major == "computer_science"

    @patch("services.job_query_rewrite_service.call_llm")
    def test_search_query_populated(self, mock_llm):
        mock_llm.return_value = (VALID_LLM_OUTPUT, {})
        query, _ = rewrite_merged_context(MERGED_CONTEXT)

        assert len(query.search_query) > 0

    @patch("services.job_query_rewrite_service.call_llm")
    def test_minimal_filters(self, mock_llm):
        mock_llm.return_value = (MINIMAL_LLM_OUTPUT, {})
        query, _ = rewrite_merged_context(MERGED_CONTEXT)

        assert query.hard_filters.min_yoe is None
        assert query.hard_filters.required_skills == []
        assert query.hard_filters.education_level is None
        assert query.hard_filters.major is None

    @patch("services.job_query_rewrite_service.call_llm")
    def test_fenced_json_output(self, mock_llm):
        mock_llm.return_value = (f"```json\n{VALID_LLM_OUTPUT}\n```", {})
        query, _ = rewrite_merged_context(MERGED_CONTEXT)

        assert query.hard_filters.min_yoe == 3

    @patch("services.job_query_rewrite_service.call_llm")
    def test_provider_and_model_forwarded(self, mock_llm):
        mock_llm.return_value = (VALID_LLM_OUTPUT, {})
        rewrite_merged_context(MERGED_CONTEXT, provider="openai", model="gpt-4o")

        _, kwargs = mock_llm.call_args
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-4o"


class TestRewriteErrors:
    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            rewrite_merged_context("")

    def test_whitespace_input_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            rewrite_merged_context("   \n\t  ")

    @patch("services.job_query_rewrite_service.call_llm")
    def test_invalid_json_raises_llm_parse_error(self, mock_llm):
        mock_llm.return_value = ("not json at all", {})
        with pytest.raises(LLMParseError, match="not valid JSON"):
            rewrite_merged_context(MERGED_CONTEXT)

    @patch("services.job_query_rewrite_service.call_llm")
    def test_schema_mismatch_raises_llm_parse_error(self, mock_llm):
        mock_llm.return_value = (json.dumps({"wrong": "shape"}), {})
        with pytest.raises(LLMParseError, match="StandardizedJobQuery"):
            rewrite_merged_context(MERGED_CONTEXT)

    @patch("services.job_query_rewrite_service.call_llm")
    def test_empty_search_query_raises_llm_parse_error(self, mock_llm):
        output = json.dumps({
            "hard_filters": {
                "min_yoe": None,
                "required_skills": [],
                "education_level": None,
                "major": None,
            },
            "search_query": "   ",
        })
        mock_llm.return_value = (output, {})
        with pytest.raises(LLMParseError, match="empty search_query"):
            rewrite_merged_context(MERGED_CONTEXT)


class TestSearchQueryEmbedding:
    @patch("services.job_query_rewrite_service.embed_text", return_value=[0.1, 0.2, 0.3])
    def test_embed_search_query_uses_same_embedding_model(self, mock_embed):
        query = StandardizedJobQuery.model_validate(json.loads(VALID_LLM_OUTPUT))
        embedding = embed_search_query(query)

        assert embedding == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(query.search_query)
