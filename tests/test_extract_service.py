import json

import services.extract_service as extract_service
from schemas.models import ExtractionInput


def test_extract_json_from_fenced_block():
    raw = """```json
{"name":"A","skills":["Python"]}
```"""
    assert extract_service._extract_json(raw) == {"name": "A", "skills": ["Python"]}


def test_extract_structured_resume_uses_schema(monkeypatch):
    def fake_call_llm(prompt: str, provider=None, model=None) -> str:
        return json.dumps(
            {
                "name": "Test",
                "email": None,
                "phone": None,
                "location": None,
                "summary": None,
                "skills": ["Python"],
                "education": [],
                "experience": [],
                "projects": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(extract_service, "call_llm", fake_call_llm)
    text = (
        "Education: University. "
        "Work experience: Internship. "
        "Skills: Python. "
        "Projects: something. "
        "Summary: candidate profile. "
        "Extra " * 20
    )
    structured = extract_service.extract_structured_resume(ExtractionInput(text=text))
    assert structured.name == "Test"
    assert structured.skills == ["Python"]
