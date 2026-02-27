import json
import re
from typing import Any, Dict

from pydantic import ValidationError
from schemas.models import ResumeStructured, ResumeSchema
from services.llm_service import call_llm
from utils.errors import LLMParseError, LLMError

def clean_and_parse_llm_output(raw_output: str) -> ResumeSchema:
    """
    Cleans the LLM output and attempts to validate it against the Pydantic schema.
    """
    # Step 1: Strip markdown formatting (e.g., ```json ... ```)
    cleaned_text = re.sub(r"```json\s*", "", raw_output)
    cleaned_text = re.sub(r"```\s*", "", cleaned_text)
    
    try:
        # Step 2: Attempt to parse string to JSON dict
        parsed_dict = json.loads(cleaned_text)
        
        # Step 3: Enforce strict schema validation using Pydantic
        # If the LLM missed a required field, this will raise a ValidationError
        valid_resume = ResumeSchema(**parsed_dict)
        return valid_resume
        
    except json.JSONDecodeError:
        # Fallback/Error handling if the LLM output is completely mangled
        raise LLMError("LLM returned malformed JSON structure.")
    except ValidationError as e:
        # Fallback/Error handling if the LLM returned JSON, but it violates our schema requirements
        raise LLMError(f"LLM output does not match required schema: {str(e)}")


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
