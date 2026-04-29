from __future__ import annotations

from services.scoring_service import ResumeScore
from routes import api


SCHEMA = {
    "schema_id": "schema-1",
    "schema_name": "DS level1",
    "version": 1,
    "summary": "Data science level 1 schema",
    "rules_json": {
        "rules1": {
            "rule_text": "Statistics fundamentals",
            "weight": 1.0,
        }
    },
}

RESUME = {
    "resume_id": "resume-1",
    "metadata": {"name": "Test Candidate"},
    "semantic_text": "Python, SQL, statistics",
    "raw_json": {"skills": ["python", "sql"]},
}


def _score(value: float, reason: str) -> ResumeScore:
    return ResumeScore.model_validate({
        "resume_id": "resume-1",
        "score": value,
        "rule_scores": {
            "rules1": {
                "score": value,
                "weight": 1.0,
                "weighted_score": value,
                "reason": reason,
            }
        },
        "explanation": reason,
    })


def test_scoring_cache_reuses_identical_input(monkeypatch):
    api._SCORING_CACHE.clear()
    monkeypatch.setattr(api.settings, "SCORING_CACHE_ENABLED", True)
    monkeypatch.setattr(api.settings, "SCORING_CACHE_MAX_SIZE", 8)

    calls = []

    def fake_score_resume_with_schema(
        *, schema, feedback_examples, resume, feedback_influence_mode, resume_evidence_by_id=None
    ):
        calls.append((schema, feedback_examples, resume))
        return _score(8.0, "first result"), {"total_tokens": 100}

    monkeypatch.setattr(api, "score_resume_with_schema", fake_score_resume_with_schema)

    first_score, first_usage = api._score_resume_with_schema_cached(
        schema=SCHEMA,
        feedback_examples=[],
        resume=RESUME,
    )
    second_score, second_usage = api._score_resume_with_schema_cached(
        schema=SCHEMA,
        feedback_examples=[],
        resume=RESUME,
    )

    assert len(calls) == 1
    assert first_score == second_score
    assert first_usage["cached"] is False
    assert second_usage["cached"] is True
    assert first_usage["cache_key"] == second_usage["cache_key"]


def test_scoring_cache_key_changes_with_feedback(monkeypatch):
    api._SCORING_CACHE.clear()
    monkeypatch.setattr(api.settings, "SCORING_CACHE_ENABLED", True)
    monkeypatch.setattr(api.settings, "SCORING_CACHE_MAX_SIZE", 8)

    calls = []

    def fake_score_resume_with_schema(
        *, schema, feedback_examples, resume, feedback_influence_mode, resume_evidence_by_id=None
    ):
        calls.append((feedback_examples, feedback_influence_mode))
        return _score(7.0 + len(calls), f"result {len(calls)}"), {}

    monkeypatch.setattr(api, "score_resume_with_schema", fake_score_resume_with_schema)

    first_score, _ = api._score_resume_with_schema_cached(
        schema=SCHEMA,
        feedback_examples=[],
        resume=RESUME,
    )
    second_score, _ = api._score_resume_with_schema_cached(
        schema=SCHEMA,
        feedback_examples=[{"feedback_id": "feedback-1", "label": "good"}],
        resume=RESUME,
    )

    assert len(calls) == 2
    assert first_score.score != second_score.score


def test_scoring_cache_key_changes_with_feedback_mode(monkeypatch):
    api._SCORING_CACHE.clear()
    monkeypatch.setattr(api.settings, "SCORING_CACHE_ENABLED", True)
    monkeypatch.setattr(api.settings, "SCORING_CACHE_MAX_SIZE", 8)

    calls = []

    def fake_score_resume_with_schema(
        *, schema, feedback_examples, resume, feedback_influence_mode, resume_evidence_by_id=None
    ):
        calls.append(feedback_influence_mode)
        return _score(6.0 + len(calls), f"mode {feedback_influence_mode}"), {}

    monkeypatch.setattr(api, "score_resume_with_schema", fake_score_resume_with_schema)

    on_score, _ = api._score_resume_with_schema_cached(
        schema=SCHEMA,
        feedback_examples=[{"feedback_id": "feedback-1", "label": "good"}],
        resume=RESUME,
        feedback_influence_mode="on",
    )
    off_score, _ = api._score_resume_with_schema_cached(
        schema=SCHEMA,
        feedback_examples=[{"feedback_id": "feedback-1", "label": "good"}],
        resume=RESUME,
        feedback_influence_mode="off",
    )

    assert calls == ["on", "off"]
    assert on_score.score != off_score.score
