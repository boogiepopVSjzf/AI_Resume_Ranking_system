from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

import certifi
import requests
from pydantic import BaseModel, ValidationError
from requests.adapters import HTTPAdapter

from config import settings
from utils.errors import LLMError


SUPPORTED_PROVIDERS = {"dashscope", "gemini", "openai", "ollama"}

# This is a workaround for ancient MacOS LibreSSL versions that cause SSL issues.
CIPHERS = (
    "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:"
    "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:"
    "ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:"
    "DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384"
)

T = TypeVar("T", bound=BaseModel)


class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = requests.packages.urllib3.util.ssl_.create_urllib3_context(ciphers=CIPHERS)
        context.load_verify_locations(certifi.where())
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = requests.packages.urllib3.util.ssl_.create_urllib3_context(ciphers=CIPHERS)
        context.load_verify_locations(certifi.where())
        kwargs["ssl_context"] = context
        return super().proxy_manager_for(*args, **kwargs)


_HTTPS_SESSION: Optional[requests.Session] = None


def _build_https_session() -> requests.Session:
    global _HTTPS_SESSION
    if _HTTPS_SESSION is None:
        session = requests.Session()
        session.mount("https://", SSLAdapter())
        session.verify = certifi.where()
        _HTTPS_SESSION = session
    return _HTTPS_SESSION


def _resolve_provider(provider: Optional[str]) -> str:
    resolved = (provider or settings.DEFAULT_LLM_PROVIDER).lower().strip()
    if resolved not in SUPPORTED_PROVIDERS:
        raise LLMError(f"Unsupported provider: {resolved}")
    return resolved


def _resolve_model(provider: str, model: Optional[str]) -> str:
    if model:
        return model

    if provider == "dashscope":
        return settings.LLM_MODEL
    if provider == "gemini":
        return settings.GEMINI_MODEL
    if provider == "openai":
        return settings.OPENAI_MODEL
    if provider == "ollama":
        return settings.OLLAMA_MODEL

    raise LLMError(f"Unsupported provider: {provider}")


def _call_gemini(prompt: str, model: str) -> tuple[str, dict]:
    if not settings.GEMINI_API_KEY:
        raise LLMError("Missing GEMINI_API_KEY")

    url = settings.GEMINI_API_URL_TEMPLATE.format(model=model)
    url = f"{url}?key={settings.GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    try:
        session = _build_https_session()
        response = session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"Gemini API error: {response.status_code} - {response.text}")

    data = response.json()

    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usage", {})
        return content, usage
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Gemini response structure: {exc}") from exc


def _call_openai(prompt: str, model: str) -> tuple[str, dict]:
    if not settings.OPENAI_API_KEY:
        raise LLMError("Missing OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
    }

    try:
        session = _build_https_session()
        response = session.post(
            settings.OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"OpenAI request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"OpenAI API error: {response.status_code} - {response.text}")

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected OpenAI response structure: {exc}") from exc


def _call_ollama(prompt: str, model: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            settings.OLLAMA_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"Ollama API error: {response.status_code} - {response.text}")

    data = response.json()

    try:
        content = data["response"]
        usage = data.get("usage", {})
        return content, usage
    except KeyError as exc:
        raise LLMError(f"Unexpected Ollama response structure: {exc}") from exc


def _call_dashscope(prompt: str, model: str) -> tuple[str, dict]:
    if not settings.LLM_API_KEY:
        raise LLMError("Missing LLM_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
    }

    try:
        session = _build_https_session()
        response = session.post(
            f"{settings.LLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Dashscope request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"Dashscope API error: {response.status_code} - {response.text}")

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Dashscope response structure: {exc}") from exc


def call_llm(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Unified LLM entrypoint.
    """
    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model(resolved_provider, model)

    if resolved_provider == "dashscope":
        return _call_dashscope(prompt, resolved_model)
    if resolved_provider == "gemini":
        return _call_gemini(prompt, resolved_model)
    if resolved_provider == "openai":
        return _call_openai(prompt, resolved_model)
    if resolved_provider == "ollama":
        return _call_ollama(prompt, resolved_model)

    raise LLMError(f"Unsupported provider: {resolved_provider}")


def _strip_code_fences(text: str) -> str:
    """
    Remove common markdown code fences around JSON output.
    """
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def _extract_first_json_block(text: str) -> str:
    """
    Extract the first JSON object or array from a text blob.
    This is a defensive fallback when the model returns extra prose.
    """
    cleaned = _strip_code_fences(text)

    object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    array_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)

    if object_match and array_match:
        if object_match.start() < array_match.start():
            return object_match.group(0)
        return array_match.group(0)

    if object_match:
        return object_match.group(0)

    if array_match:
        return array_match.group(0)

    return cleaned


def parse_llm_json_text(raw_text: str) -> Any:
    """
    Convert raw LLM text into a Python object.
    """
    candidate = _extract_first_json_block(raw_text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(
            "LLM did not return valid JSON. "
            f"Raw response begins with: {raw_text[:300]}"
        ) from exc


def build_structured_output_prompt(
    *,
    task_instruction: str,
    output_schema: dict[str, Any],
    input_sections: dict[str, Any],
) -> str:
    """
    Build a strict JSON-only prompt for structured generation.
    """
    sections = []
    for key, value in input_sections.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, indent=2)
        sections.append(f"{key}:\n{rendered}")

    joined_sections = "\n\n".join(sections)

    return f"""
You are a structured-output engine.

Task:
{task_instruction}

Rules:
1. Return valid JSON only.
2. Do not add markdown code fences.
3. Do not add explanation outside JSON.
4. Do not invent facts not supported by the input.
5. If a field cannot be determined, use null for scalar fields and [] for list fields when appropriate.
6. Follow the output schema exactly.

Output JSON schema:
{json.dumps(output_schema, ensure_ascii=False, indent=2)}

Input:
{joined_sections}
""".strip()


class LLMStructuredClient:
    """
    A small wrapper that makes the existing call_llm(...) usable for Pydantic-based
    structured output tasks such as scoring and explanation generation.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model

    def generate_text(
        self,
        prompt: str,
    ) -> tuple[str, dict]:
        return call_llm(
            prompt=prompt,
            provider=self.provider,
            model=self.model,
        )

    def generate_json(
        self,
        prompt: str,
    ) -> tuple[Any, dict]:
        raw_text, usage = self.generate_text(prompt)
        parsed = parse_llm_json_text(raw_text)
        return parsed, usage

    def generate_structured(
        self,
        *,
        task_instruction: str,
        response_model: Type[T],
        input_sections: dict[str, Any],
    ) -> tuple[T, dict]:
        prompt = build_structured_output_prompt(
            task_instruction=task_instruction,
            output_schema=response_model.model_json_schema(),
            input_sections=input_sections,
        )

        parsed_json, usage = self.generate_json(prompt)

        try:
            validated = response_model.model_validate(parsed_json)
        except ValidationError as exc:
            raise LLMError(
                "Structured output validation failed. "
                f"Expected model: {response_model.__name__}. "
                f"Validation error: {exc}"
            ) from exc

        return validated, usage