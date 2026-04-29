from __future__ import annotations

from services.scoring_service import (
    build_feedback_calibration_data,
    build_resume_scoring_prompt,
)

_EXAMPLES = [
    {
        "feedback_id": "excellent-1",
        "resume_id": "resume-1",
        "label": "excellent",
        "feedback_text": "Very strong evidence.",
        "score": 9.0,
        "scoring_result": {"score": 8.5, "rule_scores": {"rules1": {"score": 9}}},
        "created_at": "2026-01-02T00:00:00",
    },
    {
        "feedback_id": "bad-1",
        "resume_id": "resume-2",
        "label": "bad",
        "feedback_text": "Tools listed without evidence.",
        "score": 3.0,
        "created_at": "2026-01-01T00:00:00",
    },
    {
        "feedback_id": "na-1",
        "resume_id": "resume-3",
        "label": "n/a",
        "feedback_text": "",
        "score": 5.0,
        "created_at": "2026-01-03T00:00:00",
    },
]

_EVIDENCE = {
    "resume-1": {
        "resume_id": "resume-1",
        "metadata": {"role": "eng"},
        "semantic_text": "Full body for resume-1",
    },
    "resume-2": {
        "resume_id": "resume-2",
        "metadata": {},
        "semantic_text": "Body two",
    },
    "resume-3": {
        "resume_id": "resume-3",
        "metadata": {},
        "semantic_text": "",
    },
}


def test_build_feedback_calibration_data_few_shot_resume_shape():
    calibration = build_feedback_calibration_data(
        _EXAMPLES,
        resume_evidence_by_id=_EVIDENCE,
    )

    rows = calibration["few_shot_examples"]
    assert len(rows) == 3
    by_id = {(row["human_judgment"] or {})["feedback_id"]: row for row in rows}
    ex = by_id["excellent-1"]
    assert ex["resume"]["resume_id"] == "resume-1"
    assert ex["resume"]["metadata"] == {"role": "eng"}
    assert ex["resume"]["semantic_text"] == "Full body for resume-1"
    assert "raw_json" not in ex["resume"]
    assert ex["human_judgment"]["human_label"] == "excellent"
    assert "audit-only" in by_id["na-1"]["human_judgment"]["label_meaning"]
    for row in rows:
        assert row["human_judgment"]["same_resume_as_target"] is False


def test_same_resume_as_target_flag_and_ordering():
    calibration = build_feedback_calibration_data(
        _EXAMPLES,
        target_resume_id="resume-1",
        resume_evidence_by_id=_EVIDENCE,
    )
    rows = calibration["few_shot_examples"]
    assert rows[0]["human_judgment"]["feedback_id"] == "excellent-1"
    assert rows[0]["human_judgment"]["same_resume_as_target"] is True
    assert all(not r["human_judgment"]["same_resume_as_target"] for r in rows[1:])


def test_same_resume_excellent_injects_human_tier_alignment_block():
    schema = {
        "schema_id": "s1",
        "schema_name": "Test",
        "summary": "",
        "rules_json": {"rules1": {"weight": 1.0, "description": "d"}},
    }
    examples = [
        {
            "feedback_id": "fb-a",
            "resume_id": "rid-target",
            "label": "excellent",
            "feedback_text": "",
            "score": 4.85,
            "created_at": "2026-01-02T00:00:00",
        },
    ]
    evidence = {
        "rid-target": {
            "resume_id": "rid-target",
            "metadata": {},
            "semantic_text": "body",
        },
    }
    resume = {"resume_id": "rid-target", "metadata": {}, "semantic_text": "body"}
    prompt = build_resume_scoring_prompt(
        schema=schema,
        feedback_examples=examples,
        resume=resume,
        feedback_influence_mode="on",
        resume_evidence_by_id=evidence,
    )
    assert "Human tier alignment" in prompt
    assert "7.5" in prompt and "10.0" in prompt
    assert "`human_label` = `excellent`" in prompt
