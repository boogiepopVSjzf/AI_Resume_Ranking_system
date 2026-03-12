from __future__ import annotations

import requests
from typing import Optional

from config import settings
from utils.errors import LLMError


SUPPORTED_PROVIDERS = {"dashscope", "gemini", "openai", "ollama"}


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


def _call_gemini(prompt: str, model: str) -> str:
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
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
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
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Gemini response structure: {exc}") from exc


def _call_openai(prompt: str, model: str) -> str:
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
        response = requests.post(
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected OpenAI response structure: {exc}") from exc


def _call_ollama(prompt: str, model: str) -> str:
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
        return data["response"]
    except KeyError as exc:
        raise LLMError(f"Unexpected Ollama response structure: {exc}") from exc


def _call_dashscope(prompt: str, model: str) -> str:
    """Call Aliyun Dashscope API"""
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
        response = requests.post(
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Dashscope response structure: {exc}") from exc


def call_llm(prompt: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Unified LLM entrypoint.
    This function routes the request to different providers using one interface.
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