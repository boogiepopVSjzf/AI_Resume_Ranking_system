from __future__ import annotations

from unittest.mock import patch

from schemas.models import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    ResumeStructured,
)
from services.resume_storage_bundle import build_resume_storage_bundle

FAKE_EMBEDDING = [0.1, 0.2, 0.3]


def _full_resume() -> ResumeStructured:
    return ResumeStructured(
        name="张三",
        email="zhangsan@example.com",
        phone="13800138000",
        YoE="5",
        highest_education_level="硕士",
        location="上海",
        summary="资深后端工程师，擅长分布式系统。",
        skills=["Python", "Go", "SQL"],
        education=[
            EducationItem(
                school="复旦大学",
                degree="硕士",
                major="计算机科学",
                start_date="2016-09",
                end_date="2019-06",
                description="研究方向为自然语言处理，发表 SCI 论文两篇。",
            ),
        ],
        experience=[
            ExperienceItem(
                company="Google",
                title="Software Engineer",
                location="Shanghai",
                start_date="2019-07",
                end_date="2024-01",
                highlights=[
                    "Built distributed caching layer serving 10M QPS.",
                    "Led migration from monolith to microservices.",
                ],
            ),
        ],
        projects=[
            ProjectItem(
                name="ResumeBot",
                role="Tech Lead",
                start_date="2023-01",
                end_date="2023-06",
                highlights=[
                    "Designed RAG pipeline for resume matching.",
                    "Achieved 95% recall on benchmark dataset.",
                ],
            ),
        ],
    )


@patch("services.resume_storage_bundle.embed_text", return_value=FAKE_EMBEDDING)
class TestBundleShape:
    def test_keys(self, _mock_embed):
        bundle = build_resume_storage_bundle(_full_resume())
        assert set(bundle.keys()) == {"metadata", "semantic_text", "embedding", "raw_json"}

    def test_raw_json_matches_model_dump(self, _mock_embed):
        resume = _full_resume()
        bundle = build_resume_storage_bundle(resume)
        assert bundle["raw_json"] == resume.model_dump()

    def test_embedding_present(self, _mock_embed):
        bundle = build_resume_storage_bundle(_full_resume())
        assert bundle["embedding"] == FAKE_EMBEDDING

    def test_metadata_scalar_fields(self, _mock_embed):
        bundle = build_resume_storage_bundle(_full_resume())
        meta = bundle["metadata"]
        assert meta["name"] == "张三"
        assert meta["email"] == "zhangsan@example.com"
        assert meta["YoE"] == "5"

    def test_semantic_text_contains_summary(self, _mock_embed):
        bundle = build_resume_storage_bundle(_full_resume())
        assert "资深后端工程师" in bundle["semantic_text"]

    def test_semantic_text_contains_highlights(self, _mock_embed):
        bundle = build_resume_storage_bundle(_full_resume())
        assert "Built distributed caching layer" in bundle["semantic_text"]
        assert "Designed RAG pipeline" in bundle["semantic_text"]


@patch("services.resume_storage_bundle.embed_text", return_value=None)
class TestEmptySemanticText:
    def test_empty_resume_gives_none_embedding(self, _mock_embed):
        bundle = build_resume_storage_bundle(ResumeStructured())
        assert bundle["embedding"] is None
        assert bundle["semantic_text"] == ""
        assert bundle["metadata"] == {}

    def test_raw_json_still_populated(self, _mock_embed):
        bundle = build_resume_storage_bundle(ResumeStructured())
        assert bundle["raw_json"] == ResumeStructured().model_dump()
