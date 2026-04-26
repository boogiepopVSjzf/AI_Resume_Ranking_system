from __future__ import annotations

from services.scoring_service import build_feedback_calibration_data


def test_build_feedback_calibration_data_groups_labels():
    calibration = build_feedback_calibration_data([
        {
            "feedback_id": "excellent-1",
            "resume_id": "resume-1",
            "label": "excellent",
            "feedback_text": "Very strong evidence.",
            "score": 9.0,
            "scoring_result": {"score": 8.5, "rule_scores": {"rules1": {"score": 9}}},
        },
        {
            "feedback_id": "bad-1",
            "resume_id": "resume-2",
            "label": "bad",
            "feedback_text": "Tools listed without evidence.",
            "score": 3.0,
        },
        {
            "feedback_id": "na-1",
            "resume_id": "resume-3",
            "label": "n/a",
            "feedback_text": "",
            "score": 5.0,
        },
    ])

    assert calibration["positive_examples"][0]["feedback_id"] == "excellent-1"
    assert calibration["negative_examples"][0]["feedback_id"] == "bad-1"
    assert calibration["neutral_unrated_examples"][0]["feedback_id"] == "na-1"
    assert "must not calibrate scoring" in calibration["neutral_unrated_examples"][0]["label_meaning"]
