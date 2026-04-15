from __future__ import annotations

from schemas.final_result import RuleSchemaResponse
from storage import rule_schema_store
from storage.rule_schema_store import (
    _embedding_dimension,
    _ensure_rule_schema_dataset,
    _vector_literal,
    save_rule_schema_result,
)


class FakeCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.calls.append((query, params))


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_vector_literal_formats_pgvector_text():
    assert _vector_literal([0.1, 2, -3.5]) == "[0.1,2.0,-3.5]"


def test_embedding_dimension_uses_vector_length():
    assert _embedding_dimension([1.0, 2.0, 3.0]) == 3


def test_ensure_rule_schema_dataset_creates_table(monkeypatch):
    monkeypatch.setattr(rule_schema_store.settings, "DB_AUTO_INIT", True)
    monkeypatch.setattr(rule_schema_store, "_RULE_SCHEMA_READY", False)
    conn = FakeConn()

    _ensure_rule_schema_dataset(conn, 3)

    executed_sql = "\n".join(query for query, _params in conn.cursor_obj.calls)
    assert "create table if not exists rule_schemas" in executed_sql
    assert "embedding vector(3)" in executed_sql
    assert "idx_rule_schemas_embedding_hnsw" in executed_sql
    assert conn.commits == 1


def test_save_rule_schema_result_upserts_row(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(rule_schema_store, "_connect", lambda: conn)
    monkeypatch.setattr(rule_schema_store, "_ensure_rule_schema_dataset", lambda _conn, _dim: None)

    save_rule_schema_result(
        RuleSchemaResponse(
            schema_id="schema123",
            rule_json={"min_yoe": 3},
            rule_description="Requires at least 3 years of experience.",
            embedding_vector=[0.1, 0.2, 0.3],
            job_name="data science",
        )
    )

    query, params = conn.cursor_obj.calls[0]
    assert "insert into rule_schemas" in query
    assert "on conflict (schema_id)" in query
    assert params["schema_id"] == "schema123"
    assert params["rule_json"] == '{"min_yoe": 3}'
    assert params["embedding"] == "[0.1,0.2,0.3]"
    assert params["job_name"] == "data science"
    assert conn.commits == 1
