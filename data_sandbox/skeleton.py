"""Template skeleton builder for synthetic resume generation.

PII fields use sequential numbering; descriptive fields are left as
placeholders (None / []) for downstream LLM enrichment.

Optional fields other than name / email / phone may be set to null with
probability ``OPTIONAL_FIELD_NULL_PROB`` (lists may be empty instead).
"""
from __future__ import annotations

import random
from typing import Optional, TypeVar, get_args

from schemas.models import EducationLevel, MajorCategory

# Optional scalar fields (everything except name / email / phone) may become null
# with this probability, to diversify sparse vs dense records.
OPTIONAL_FIELD_NULL_PROB = 0.22

_T = TypeVar("_T")


def _maybe_null(value: _T, prob: float = OPTIONAL_FIELD_NULL_PROB) -> Optional[_T]:
    """Return ``None`` with probability *prob*, else *value*."""
    if random.random() < prob:
        return None
    return value


# ── Enum value lists ────────────────────────────────────────────────

EDUCATION_LEVELS: list[str] = list(get_args(EducationLevel))
MAJOR_CATEGORIES: list[str] = list(get_args(MajorCategory))

# ── Skill pools by major ───────────────────────────────────────────

SKILLS_POOL: dict[str, list[str]] = {
    "computer_science": [
        "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "SQL",
        "React", "Node.js", "Docker", "Kubernetes", "AWS", "Git", "Linux",
        "TensorFlow", "PyTorch", "PostgreSQL", "MongoDB", "Redis", "CI/CD",
    ],
    "mathematics": [
        "Python", "R", "MATLAB", "LaTeX", "SQL", "SAS",
        "NumPy", "SciPy", "Pandas", "Statistical Modeling",
        "Optimization", "Machine Learning", "Probability Theory",
    ],
    "medicine": [
        "Clinical Research", "HIPAA Compliance", "EHR Systems", "SPSS",
        "R", "Epidemiology", "Biostatistics", "GCP", "Data Analysis",
        "Medical Terminology", "Literature Review", "Patient Care",
    ],
    "finance": [
        "Python", "R", "SQL", "Excel VBA", "Bloomberg", "MATLAB",
        "Financial Modeling", "Risk Management", "Derivatives",
        "Monte Carlo", "Portfolio Optimization", "Tableau",
    ],
    "engineering": [
        "MATLAB", "AutoCAD", "SolidWorks", "Python", "C++",
        "Finite Element Analysis", "CAD/CAM", "Project Management",
        "Six Sigma", "PLC Programming", "Circuit Design",
    ],
    "other": [
        "Python", "SQL", "Excel", "Tableau", "Power BI",
        "Data Analysis", "Project Management", "Communication",
        "Problem Solving", "Agile", "Scrum",
    ],
}

# ── Entity pools ───────────────────────────────────────────────────

SCHOOLS = [
    "Massachusetts Institute of Technology",
    "Stanford University",
    "Carnegie Mellon University",
    "University of California, Berkeley",
    "Georgia Institute of Technology",
    "Columbia University",
    "University of Michigan",
    "New York University",
    "Cornell University",
    "University of Illinois at Urbana-Champaign",
    "University of Texas at Austin",
    "Purdue University",
    "University of Washington",
    "University of Wisconsin-Madison",
    "Pennsylvania State University",
]

COMPANIES: dict[str, list[str]] = {
    "computer_science": [
        "Google", "Amazon", "Meta", "Microsoft", "Apple", "Netflix",
        "Uber", "Stripe", "Airbnb", "Salesforce", "Oracle", "Adobe",
    ],
    "finance": [
        "Goldman Sachs", "JPMorgan Chase", "Morgan Stanley", "Citadel",
        "Two Sigma", "BlackRock", "Fidelity", "Bank of America",
    ],
    "medicine": [
        "Mayo Clinic", "Johns Hopkins Hospital", "Pfizer",
        "Johnson & Johnson", "Merck", "Roche", "AstraZeneca",
    ],
    "mathematics": [
        "Two Sigma", "Renaissance Technologies", "Citadel",
        "DE Shaw", "Palantir", "MathWorks", "SAS Institute",
    ],
    "engineering": [
        "Boeing", "SpaceX", "Tesla", "General Electric",
        "Siemens", "Honeywell", "Lockheed Martin", "3M",
    ],
    "other": [
        "Deloitte", "McKinsey", "Accenture", "PwC", "Amazon",
        "Google", "Microsoft", "IBM",
    ],
}

TITLES: dict[str, list[str]] = {
    "computer_science": [
        "Software Engineer", "Senior Software Engineer", "Backend Developer",
        "Full Stack Developer", "Data Engineer", "ML Engineer",
        "DevOps Engineer", "Software Development Intern",
    ],
    "finance": [
        "Quantitative Analyst", "Financial Analyst", "Risk Analyst",
        "Portfolio Manager", "Research Analyst", "Finance Intern",
    ],
    "medicine": [
        "Clinical Research Coordinator", "Medical Researcher",
        "Biostatistician", "Health Data Analyst", "Research Assistant",
    ],
    "mathematics": [
        "Quantitative Researcher", "Data Scientist", "Statistician",
        "Actuarial Analyst", "Research Assistant",
    ],
    "engineering": [
        "Mechanical Engineer", "Electrical Engineer", "Systems Engineer",
        "Design Engineer", "Engineering Intern",
    ],
    "other": [
        "Business Analyst", "Consultant", "Project Manager",
        "Operations Analyst", "Product Manager",
    ],
}

LOCATIONS = [
    "New York, NY", "San Francisco, CA", "Seattle, WA", "Boston, MA",
    "Austin, TX", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
    "Atlanta, GA", "Washington, DC", "Remote",
]

DEGREE_NAMES: dict[str, str] = {
    "high_school": "High School Diploma",
    "associate": "Associate Degree",
    "bachelor": "Bachelor of Science",
    "master": "Master of Science",
    "phd": "Doctor of Philosophy",
    "other": "Professional Certificate",
}

MAJOR_TEXTS: dict[str, list[str]] = {
    "computer_science": ["Computer Science", "Software Engineering", "Information Systems"],
    "mathematics": ["Mathematics", "Applied Mathematics", "Statistics"],
    "medicine": ["Medicine", "Biomedical Sciences", "Public Health"],
    "finance": ["Finance", "Financial Engineering", "Economics"],
    "engineering": ["Mechanical Engineering", "Electrical Engineering", "Civil Engineering"],
    "other": ["Business Administration", "Liberal Arts", "Communications"],
}

PROJECT_ROLES: list[str | None] = [
    "Lead Developer", "Team Lead", "Contributor", "Researcher", None,
]

# ── Tier parameters ────────────────────────────────────────────────

# YoE strings: years only (no months), so parse_yoe_num in resume_rag_split matches human intent.
TIER_PARAMS: dict[str, dict] = {
    "minimal": {
        "edu_range": (0, 1),
        "exp_range": (0, 1),
        "proj_range": (0, 0),
        "skill_range": (1, 3),
        "yoe_choices": ["0 years", "1 year", "2 years"],
    },
    "typical": {
        "edu_range": (1, 2),
        "exp_range": (1, 3),
        "proj_range": (0, 2),
        "skill_range": (4, 8),
        "yoe_choices": ["1 year", "2 years", "3 years", "5 years"],
    },
    "complex": {
        "edu_range": (2, 3),
        "exp_range": (3, 5),
        "proj_range": (2, 4),
        "skill_range": (8, 15),
        "yoe_choices": ["6 years", "8 years", "10 years", "15 years"],
    },
}

TIERS = list(TIER_PARAMS.keys())

# ── Helpers ─────────────────────────────────────────────────────────


def _sample(pool: list, k: int) -> list:
    if k >= len(pool):
        return list(pool)
    return random.sample(pool, k)


def _random_date_range() -> tuple[str, str]:
    """Return (start_date, end_date) in MM/YY format."""
    start_year = random.randint(2016, 2024)
    start_month = random.randint(1, 12)
    duration = random.randint(3, 30)
    end_month = start_month + duration
    end_year = start_year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    return (
        f"{start_month:02d}/{start_year % 100:02d}",
        f"{end_month:02d}/{end_year % 100:02d}",
    )


def _random_end_date() -> str:
    y = random.randint(2018, 2025)
    m = random.randint(1, 12)
    return f"{m:02d}/{y % 100:02d}"


# ── Public builder ──────────────────────────────────────────────────


def build_skeleton(index: int, tier: str) -> dict:
    """Build a raw_json skeleton for resume *index* at *tier* complexity.

    Sequential PII fields; descriptive fields left as ``None`` / ``[]``
    for LLM enrichment.
    """
    params = TIER_PARAMS[tier]
    # Internal draws (used for pools); root fields may be nulled independently below.
    education_level = random.choice(EDUCATION_LEVELS)
    major = random.choice(MAJOR_CATEGORIES)
    skill_pool = SKILLS_POOL.get(major, SKILLS_POOL["other"])
    n_skills = random.randint(*params["skill_range"])
    skills: list[str] = (
        [] if random.random() < OPTIONAL_FIELD_NULL_PROB else _sample(skill_pool, n_skills)
    )

    skeleton: dict = {
        "name": f"Candidate {index:03d}",
        "email": f"candidate{index:03d}@example.com",
        "phone": f"(000) 000-{index:04d}",
        "YoE": _maybe_null(random.choice(params["yoe_choices"])),
        "education_level": _maybe_null(education_level),
        "major": _maybe_null(major),
        "location": _maybe_null(random.choice(LOCATIONS)),
        "skills": skills,
        "education": [],
        "experience": [],
        "projects": [],
    }

    major_texts = MAJOR_TEXTS.get(major, MAJOR_TEXTS["other"])
    for _ in range(random.randint(*params["edu_range"])):
        start_cand = None if random.random() < 0.72 else _random_end_date()
        skeleton["education"].append({
            "school": _maybe_null(random.choice(SCHOOLS)),
            "degree": _maybe_null(DEGREE_NAMES.get(education_level, "Bachelor of Science")),
            "major": _maybe_null(random.choice(major_texts)),
            "start_date": start_cand if start_cand is None else _maybe_null(start_cand),
            "end_date": _maybe_null(_random_end_date()),
            "description": None,
        })

    companies = COMPANIES.get(major, COMPANIES["other"])
    titles = TITLES.get(major, TITLES["other"])
    for _ in range(random.randint(*params["exp_range"])):
        start, end = _random_date_range()
        skeleton["experience"].append({
            "company": _maybe_null(random.choice(companies)),
            "title": _maybe_null(random.choice(titles)),
            "location": _maybe_null(random.choice(LOCATIONS)),
            "start_date": _maybe_null(start),
            "end_date": _maybe_null(end),
            "highlights": [],
        })

    for _ in range(random.randint(*params["proj_range"])):
        start, end = _random_date_range()
        role = random.choice(PROJECT_ROLES)
        skeleton["projects"].append({
            "name": None,
            "role": _maybe_null(role),
            "start_date": _maybe_null(start),
            "end_date": _maybe_null(end),
            "highlights": [],
        })

    return skeleton
