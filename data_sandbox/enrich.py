"""LLM enrichment and schema validation / repair for synthetic resumes."""
from __future__ import annotations

from typing import Optional

from pydantic import ValidationError

from schemas.models import ResumeStructured
from services.llm_service import call_llm
from utils.llm_json import extract_json
from utils.logger import get_logger

logger = get_logger("sandbox.enrich")

_VALID_EDUCATION_LEVELS = {
    "high_school", "associate", "bachelor", "master", "phd", "other",
}
_VALID_MAJORS = {
    "computer_science", "mathematics", "medicine", "finance", "engineering", "other",
}


# ── LLM enrichment ─────────────────────────────────────────────────


def _needs_llm(skeleton: dict) -> bool:
    return bool(
        skeleton.get("education")
        or skeleton.get("experience")
        or skeleton.get("projects")
    )


def _build_enrich_prompt(skeleton: dict) -> str:
    major = skeleton.get("major", "other")
    skills_csv = ", ".join(skeleton.get("skills", []))

    sections: list[str] = []

    edu = skeleton.get("education", [])
    if edu:
        lines = [
            f"  {i + 1}. {e.get('degree')} in {e.get('major')} at {e.get('school')}"
            for i, e in enumerate(edu)
        ]
        sections.append(
            f"Education ({len(edu)}):\n" + "\n".join(lines)
            + "\n→ Generate a brief description for each (GPA, honors, coursework)"
        )

    exp = skeleton.get("experience", [])
    if exp:
        lines = [
            f"  {i + 1}. {e.get('title')} at {e.get('company')}"
            for i, e in enumerate(exp)
        ]
        sections.append(
            f"Experience ({len(exp)}):\n" + "\n".join(lines)
            + "\n→ Generate 2-4 professional highlight bullets per entry"
        )

    proj = skeleton.get("projects", [])
    if proj:
        lines = [
            f"  {i + 1}. Role: {p.get('role') or 'Contributor'}"
            for i, p in enumerate(proj)
        ]
        sections.append(
            f"Projects ({len(proj)}):\n" + "\n".join(lines)
            + "\n→ Generate a technical project name and 2-3 highlight bullets per entry"
        )

    keys: list[str] = []
    if edu:
        keys.append('"edu_desc": ["...", ...]')
    if exp:
        keys.append('"exp_hl": [["bullet", ...], ...]')
    if proj:
        keys.append('"proj_names": ["...", ...]')
        keys.append('"proj_hl": [["bullet", ...], ...]')
    json_shape = "{" + ", ".join(keys) + "}"

    return (
        "Generate realistic English resume content. Return ONLY valid JSON, no markdown.\n\n"
        f"Field: {major} | Skills: {skills_csv}\n\n"
        + "\n\n".join(sections)
        + f"\n\nJSON: {json_shape}"
    )


def _merge_llm_output(skeleton: dict, llm_data: dict) -> dict:
    result = {k: (list(v) if isinstance(v, list) else v) for k, v in skeleton.items()}
    result["education"] = [dict(e) for e in skeleton.get("education", [])]
    result["experience"] = [dict(e) for e in skeleton.get("experience", [])]
    result["projects"] = [dict(p) for p in skeleton.get("projects", [])]

    for i, entry in enumerate(result["education"]):
        descs = llm_data.get("edu_desc", [])
        if i < len(descs) and isinstance(descs[i], str):
            entry["description"] = descs[i]

    for i, entry in enumerate(result["experience"]):
        hls = llm_data.get("exp_hl", [])
        if i < len(hls) and isinstance(hls[i], list):
            entry["highlights"] = [str(h) for h in hls[i] if h]

    proj_names = llm_data.get("proj_names", [])
    proj_hls = llm_data.get("proj_hl", [])
    for i, entry in enumerate(result["projects"]):
        if i < len(proj_names) and isinstance(proj_names[i], str):
            entry["name"] = proj_names[i]
        if i < len(proj_hls) and isinstance(proj_hls[i], list):
            entry["highlights"] = [str(h) for h in proj_hls[i] if h]

    return result


# ── Validation & repair ─────────────────────────────────────────────


def _rule_repair(data: dict) -> dict:
    """Best-effort rule-based repair to conform to ResumeStructured."""
    if data.get("education_level") not in _VALID_EDUCATION_LEVELS:
        data["education_level"] = None
    if data.get("major") not in _VALID_MAJORS:
        data["major"] = None

    for field in ("skills", "education", "experience", "projects"):
        if not isinstance(data.get(field), list):
            data[field] = []

    data["skills"] = [str(s) for s in data["skills"] if s]

    for exp in data.get("experience", []):
        if not isinstance(exp.get("highlights"), list):
            exp["highlights"] = []
        exp["highlights"] = [str(h) for h in exp["highlights"] if h]

    for proj in data.get("projects", []):
        if not isinstance(proj.get("highlights"), list):
            proj["highlights"] = []
        proj["highlights"] = [str(h) for h in proj["highlights"] if h]

    return data


# ── Public API ──────────────────────────────────────────────────────


def enrich_and_validate(
    skeleton: dict,
    *,
    max_repair: int = 2,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> ResumeStructured:
    """Enrich *skeleton* via LLM, validate against ResumeStructured, auto-repair."""
    if _needs_llm(skeleton):
        prompt = _build_enrich_prompt(skeleton)
        try:
            raw_output, _ = call_llm(prompt, provider=provider, model=model)
            llm_data = extract_json(raw_output)
            data = _merge_llm_output(skeleton, llm_data)
        except Exception as exc:
            logger.warning("LLM enrichment failed, using skeleton: %s", exc)
            data = dict(skeleton)
    else:
        data = dict(skeleton)

    for attempt in range(max_repair + 1):
        data = _rule_repair(data)
        try:
            return ResumeStructured.model_validate(data)
        except ValidationError as exc:
            if attempt == max_repair:
                logger.error("Validation failed after %d repairs: %s", max_repair, exc)
                raise
            logger.info("Repair attempt %d for validation error", attempt + 1)

    return ResumeStructured.model_validate(data)
