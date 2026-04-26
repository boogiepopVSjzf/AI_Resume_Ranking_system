from __future__ import annotations

from routes import api


RULES = (
    "rules1: Evaluate SQL querying ability, including joins and aggregations. weight: 50% "
    "rules2: Evaluate Python data science execution. weight: 50%"
)


def test_schema_cache_reuses_identical_schema_input(monkeypatch):
    api._SCHEMA_CACHE.clear()
    monkeypatch.setattr(api.settings, "SCHEMA_CACHE_ENABLED", True)
    monkeypatch.setattr(api.settings, "SCHEMA_CACHE_MAX_SIZE", 8)

    build_calls = []
    save_calls = []

    def fake_build_scoring_schema_payload(schema_name, rules):
        build_calls.append((schema_name, rules))
        return {
            "schema_name": schema_name,
            "rules_json": api.parse_rules_text(rules),
            "summary": "Stable schema summary.",
            "embedding": [0.1, 0.2, 0.3],
        }, {"total_tokens": 100}

    def fake_save_scoring_schema(**kwargs):
        save_calls.append(kwargs)
        return {
            "schema_id": "schema-1",
            "schema_name": kwargs["schema_name"],
            "rules_json": kwargs["rules_json"],
            "summary": kwargs["summary"],
            "version": 1,
            "is_active": True,
        }

    monkeypatch.setattr(api, "build_scoring_schema_payload", fake_build_scoring_schema_payload)
    monkeypatch.setattr(api, "save_scoring_schema", fake_save_scoring_schema)

    first = api._create_scoring_schema_cached(
        schema_name="Data Science Level 1",
        rules=RULES,
    )
    second = api._create_scoring_schema_cached(
        schema_name="Data Science Level 1",
        rules=RULES,
    )

    assert len(build_calls) == 1
    assert len(save_calls) == 1
    assert first["schema_id"] == second["schema_id"]
    assert first["version"] == second["version"]
    assert first["usage"]["cached"] is False
    assert second["usage"]["cached"] is True
    assert second["schema_cache_reused"] is True
