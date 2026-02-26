import json
import re
from typing import Any, Dict

from schemas.models import ResumeStructured
from services.llm_service import call_llm
from utils.errors import LLMParseError


def build_prompt(text: str) -> str:
    schema_hint = {
        "name": "string or null",
        "email": "string or null",
        "phone": "string or null",
        "location": "string or null",
        "summary": "string or null",
        "skills": ["string"],
        "education": [
            {
                "school": "string or null",
                "degree": "string or null",
                "major": "string or null",
                "start_date": "string or null",
                "end_date": "string or null",
                "description": "string or null",
            }
        ],
        "experience": [
            {
                "company": "string or null",
                "title": "string or null",
                "location": "string or null",
                "start_date": "string or null",
                "end_date": "string or null",
                "highlights": ["string"],
            }
        ],
        "projects": [
            {
                "name": "string or null",
                "role": "string or null",
                "start_date": "string or null",
                "end_date": "string or null",
                "highlights": ["string"],
            }
        ],
    }
    return (
        "你是一个简历解析器。请把输入简历文本转换为JSON。"
        "只输出JSON，不要解释。字段必须匹配以下结构：\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}\n\n简历文本：\n{text}"
    )


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise LLMParseError("LLM 输出不是有效 JSON")


def extract_structured_resume(text: str) -> ResumeStructured:
    prompt = build_prompt(text)
    raw = call_llm(prompt)
    data = _extract_json(raw)
    return ResumeStructured.model_validate(data)
