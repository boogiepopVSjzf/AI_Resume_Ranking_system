from __future__ import annotations

from schemas.models import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    ResumeStructured,
)
from services.resume_rag_split import normalize_major, split_for_rag


def _full_resume() -> ResumeStructured:
    return ResumeStructured(
        name="张三",
        email="zhangsan@example.com",
        phone="13800138000",
        YoE="5",
        education_level="master",
        major="computer_science",
        location="上海",
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


class TestMetadata:
    def test_scalar_fields_present(self):
        result = split_for_rag(_full_resume())
        meta = result["metadata"]
        assert meta["name"] == "张三"
        assert meta["email"] == "zhangsan@example.com"
        assert meta["phone"] == "13800138000"
        assert meta["yoe_text"] == "5"
        assert meta["yoe_num"] == 5.0
        assert meta["education_level"] == "master"
        assert meta["major"] == "computer_science"

    def test_skills_present(self):
        result = split_for_rag(_full_resume())
        assert result["metadata"]["skills"] == ["python", "go", "sql"]

    def test_education_meta_excludes_description(self):
        result = split_for_rag(_full_resume())
        edu = result["metadata"]["education"]
        assert len(edu) == 1
        assert edu[0]["school"] == "复旦大学"
        assert edu[0]["degree"] == "硕士"
        assert edu[0]["major"] == "计算机科学"
        assert "description" not in edu[0]

    def test_no_experience_or_project_in_metadata(self):
        result = split_for_rag(_full_resume())
        meta = result["metadata"]
        assert "experience" not in meta
        assert "projects" not in meta

    def test_no_summary_in_metadata(self):
        result = split_for_rag(_full_resume())
        assert "summary" not in result["metadata"]


class TestFullText:
    def test_education_description_with_prefix(self):
        result = split_for_rag(_full_resume())
        assert "[复旦大学, 硕士] 研究方向为自然语言处理" in result["full_text"]

    def test_experience_highlights_with_prefix(self):
        result = split_for_rag(_full_resume())
        text = result["full_text"]
        assert "[Google, Software Engineer]" in text
        assert "Built distributed caching layer" in text
        assert "Led migration from monolith" in text

    def test_project_highlights_with_prefix(self):
        result = split_for_rag(_full_resume())
        text = result["full_text"]
        assert "[ResumeBot, Tech Lead]" in text
        assert "Designed RAG pipeline" in text

    def test_paragraphs_separated_by_double_newline(self):
        result = split_for_rag(_full_resume())
        paragraphs = result["full_text"].split("\n\n")
        assert len(paragraphs) == 3


class TestEdgeCases:
    def test_empty_resume(self):
        result = split_for_rag(ResumeStructured())
        assert result["metadata"] == {}
        assert result["full_text"] == ""

    def test_none_scalars_excluded_from_metadata(self):
        resume = ResumeStructured(name="Alice")
        meta = split_for_rag(resume)["metadata"]
        assert meta == {"name": "Alice"}

    def test_education_description_without_school_degree(self):
        resume = ResumeStructured(
            education=[EducationItem(description="Self-taught ML.")]
        )
        result = split_for_rag(resume)
        assert "education" not in result["metadata"]
        assert result["full_text"] == "Self-taught ML."

    def test_experience_no_highlights_skipped(self):
        resume = ResumeStructured(
            experience=[ExperienceItem(company="Acme", title="Intern")]
        )
        result = split_for_rag(resume)
        assert result["full_text"] == ""

    def test_project_partial_prefix(self):
        resume = ResumeStructured(
            projects=[
                ProjectItem(
                    name="Foo",
                    highlights=["Did something cool."],
                )
            ]
        )
        result = split_for_rag(resume)
        assert result["full_text"] == "[Foo] Did something cool."

    def test_skills_empty_list_not_in_metadata(self):
        resume = ResumeStructured(name="Bob", skills=[])
        meta = split_for_rag(resume)["metadata"]
        assert "skills" not in meta

    def test_financial_math_major_collapses_to_finance(self):
        assert normalize_major("Financial Mathematics") == "finance"
