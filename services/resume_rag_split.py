from __future__ import annotations

from typing import Any

from schemas.models import ResumeStructured


_SCALAR_FIELDS = ("name", "email", "phone", "YoE", "highest_education_level", "location")
_EDU_META_FIELDS = ("school", "degree", "major", "start_date", "end_date")


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
        meta["skills"] = list(resume.skills)

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

    if resume.summary:
        paragraphs.append(resume.summary)

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
