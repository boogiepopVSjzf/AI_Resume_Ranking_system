from __future__ import annotations

import re
from typing import Any

from schemas.models import ResumeStructured


_SCALAR_FIELDS = ("name", "email", "phone")
_EDU_META_FIELDS = ("school", "degree", "major", "start_date", "end_date")

_MAJOR_CATEGORY_KEYWORDS = {
    "computer_science": (
        "computer science",
        "computer engineering",
        "software engineering",
        "software",
        "计算机",
        "软件工程",
        "信息工程",
        "人工智能",
        "数据科学",
    ),
    "mathematics": (
        "mathematics",
        "math",
        "applied math",
        "statistics",
        "statistical",
        "数学",
        "统计",
    ),
    "medicine": (
        "medicine",
        "medical",
        "clinical",
        "biomedical",
        "pharmacy",
        "医学",
        "临床",
        "药学",
    ),
    "finance": (
        "finance",
        "financial",
        "accounting",
        "economics",
        "fintech",
        "金融",
        "会计",
        "经济",
    ),
    "engineering": (
        "engineering",
        "electrical",
        "mechanical",
        "civil",
        "industrial",
        "工程",
        "电子",
        "机械",
        "土木",
    ),
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def normalize_major(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_text(value)
    if "finance" in normalized or "financial" in normalized or "金融" in normalized:
        return "finance"
    for category, keywords in _MAJOR_CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "other"


def normalize_skill(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_text(value)
    normalized = normalized.replace("c++", "cpp").replace("c#", "csharp")
    normalized = re.sub(r"[^a-z0-9_+#./-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or None


def parse_yoe_num(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _build_prefix(parts: list[str | None]) -> str:
    """Build a '[a, b]' prefix from non-None parts. Returns '' if all are None."""
    filled = [p for p in parts if p]
    if not filled:
        return ""
    return f"[{', '.join(filled)}] "


def _build_metadata(resume: ResumeStructured) -> dict[str, Any]:
    meta: dict[str, Any] = {}

    for field in _SCALAR_FIELDS:
        val = getattr(resume, field)
        if val is not None:
            meta[field] = val

    if resume.skills:
        normalized_skills: list[str] = []
        seen_skills: set[str] = set()
        for skill in resume.skills:
            normalized = normalize_skill(skill)
            if normalized and normalized not in seen_skills:
                seen_skills.add(normalized)
                normalized_skills.append(normalized)
        if normalized_skills:
            meta["skills"] = normalized_skills

    if resume.YoE is not None:
        meta["yoe_text"] = resume.YoE
        yoe_num = parse_yoe_num(resume.YoE)
        if yoe_num is not None:
            meta["yoe_num"] = yoe_num

    if resume.education_level is not None:
        meta["education_level"] = resume.education_level

    if resume.major is not None:
        meta["major"] = resume.major
    else:
        normalized_major = None
        for edu in resume.education:
            normalized_major = normalize_major(edu.major)
            if normalized_major:
                break
        if normalized_major is not None:
            meta["major"] = normalized_major

    edu_list: list[dict[str, Any]] = []
    for edu in resume.education:
        entry = {}
        for f in _EDU_META_FIELDS:
            val = getattr(edu, f)
            if val is not None:
                entry[f] = val
        if entry:
            edu_list.append(entry)
    if edu_list:
        meta["education"] = edu_list

    return meta


def _build_full_text(resume: ResumeStructured) -> str:
    paragraphs: list[str] = []

    for edu in resume.education:
        if edu.description:
            prefix = _build_prefix([edu.school, edu.degree])
            paragraphs.append(f"{prefix}{edu.description}")

    for exp in resume.experience:
        if exp.highlights:
            prefix = _build_prefix([exp.company, exp.title])
            paragraphs.append(f"{prefix}{' '.join(exp.highlights)}")

    for proj in resume.projects:
        if proj.highlights:
            prefix = _build_prefix([proj.name, proj.role])
            paragraphs.append(f"{prefix}{' '.join(proj.highlights)}")

    return "\n\n".join(paragraphs)


def split_for_rag(resume: ResumeStructured) -> dict[str, Any]:
    """Internal: split a ResumeStructured into metadata and full_text.

    Consumers should use ``build_resume_storage_bundle`` instead of calling
    this directly — it produces the canonical unified output.
    """
    return {
        "metadata": _build_metadata(resume),
        "full_text": _build_full_text(resume),
    }
