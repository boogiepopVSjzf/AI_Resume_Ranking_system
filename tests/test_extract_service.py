import json

import services.extract_service as extract_service


def test_extract_json_from_fenced_block():
    raw = """```json
{"name":"A","skills":["Python"]}
```"""
    assert extract_service._extract_json(raw) == {"name": "A", "skills": ["Python"]}


def test_extract_structured_resume_uses_schema(monkeypatch):
    def fake_call_llm(prompt: str) -> str:
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
    structured = extract_service.extract_structured_resume("whatever")
    assert structured.name == "Test"
    assert structured.skills == ["Python"]
