from __future__ import annotations

import pytest

from storage.postgres_store import (
    _embedding_dimension,
    _extract_skills,
    _parse_yoe_num,
    _vector_literal,
)
from utils.errors import DatabaseError


def test_vector_literal_formats_pgvector_text():
    assert _vector_literal([0.1, 2, -3.5]) == "[0.1,2.0,-3.5]"


def test_vector_literal_returns_none_for_missing_embedding():
    assert _vector_literal(None) is None


def test_vector_literal_rejects_empty_embedding():
    with pytest.raises(DatabaseError, match="empty"):
        _vector_literal([])


def test_embedding_dimension_uses_vector_length_when_present():
    assert _embedding_dimension([1.0, 2.0, 3.0]) == 3


def test_parse_yoe_num_extracts_first_numeric_value():
    assert _parse_yoe_num("3.5 years") == 3.5


def test_extract_skills_filters_blank_entries():
    assert _extract_skills({"skills": ["Python", " ", "SQL"]}) == ["Python", "SQL"]
