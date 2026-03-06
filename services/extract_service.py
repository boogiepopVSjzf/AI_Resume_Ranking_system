import json
import re
from typing import Any, Dict

from schemas.models import ResumeStructured
from services.llm_service import call_llm
from utils.errors import LLMParseError


def _schema_hint_from_model() -> Dict[str, Any]:
    schema = ResumeStructured.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve_ref(ref: str) -> Dict[str, Any]:
        if ref.startswith("#/$defs/"):
            key = ref.split("/")[-1]
            return defs.get(key, {})
        return {}

    def map_simple(type_name: str, optional: bool) -> str:
        if optional:
            return f"{type_name} or null"
        return type_name

    def build(node: Any) -> Any:
        if not isinstance(node, dict):
            return "string"
        if "$ref" in node:
            return build(resolve_ref(node["$ref"]))
        if "anyOf" in node:
            any_of = node.get("anyOf") or []
            non_null = [
                item
                for item in any_of
                if not (isinstance(item, dict) and item.get("type") == "null")
            ]
            optional = len(non_null) != len(any_of)
            if not non_null:
                return "string or null"
            built = build(non_null[0])
            if isinstance(built, str):
                return map_simple(built.replace(" or null", ""), optional)
            return built
        node_type = node.get("type")
        if node_type == "object":
            props = node.get("properties") or {}
            return {key: build(val) for key, val in props.items()}
        if node_type == "array":
            items = node.get("items") or {}
            return [build(items)]
        if isinstance(node_type, list):
            optional = "null" in node_type
            non_null = [t for t in node_type if t != "null"]
            base = non_null[0] if non_null else "string"
            return map_simple(base, optional)
        if node_type == "string":
            return "string"
        if node_type == "integer":
            return "integer"
        if node_type == "number":
            return "number"
        if node_type == "boolean":
            return "boolean"
        return "string"

    return build(schema)


def build_prompt(text: str) -> str:
    schema_hint = _schema_hint_from_model()
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
