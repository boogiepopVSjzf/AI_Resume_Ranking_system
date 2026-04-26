from __future__ import annotations

import certifi
import requests
from requests.adapters import HTTPAdapter
from typing import Optional

from config import settings
from utils.errors import LLMError


SUPPORTED_PROVIDERS = {"dashscope", "gemini", "openai", "ollama"}

# This is a workaround for ancient MacOS LibreSSL versions that cause SSLEOFError.
# It forces requests to use a more robust set of ciphers.
# See: https://github.com/urllib3/urllib3/issues/2653
CIPHERS = (
    "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:"
    "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:"
    "DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384"
)

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


_HTTPS_SESSION = None


def _build_https_session() -> requests.Session:
    global _HTTPS_SESSION
    if _HTTPS_SESSION is None:
        session = requests.Session()
        session.mount("https://", SSLAdapter())
        session.verify = certifi.where()
        _HTTPS_SESSION = session
    return _HTTPS_SESSION

#把传入的 provider（或默认配置）规范化成小写无空格的值，并校验它必须在系统支持的 provider 列表里，否则直接报错。
def _resolve_provider(provider: Optional[str]) -> str:
    resolved = (provider or settings.DEFAULT_LLM_PROVIDER).lower().strip()
    if resolved not in SUPPORTED_PROVIDERS:
        raise LLMError(f"Unsupported provider: {resolved}")
    return resolved

#根据 provider 选择对应的 model，如果 model 为空则使用默认配置。
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

#调用 Gemini 模型
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
            "temperature": settings.LLM_TEMPERATURE,
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
        # Gemini API v1beta doesn't consistently return usage stats in this format
        usage = data.get("usage", {})
        return content, usage
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Gemini response structure: {exc}") from exc

#调用 OpenAI 模型
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
        "temperature": settings.LLM_TEMPERATURE,
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

#调用 Ollama 模型
def _call_ollama(prompt: str, model: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.LLM_TEMPERATURE,
        },
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
        # Ollama doesn't provide token usage in the same way
        usage = data.get("usage", {})
        return content, usage
    except KeyError as exc:
        raise LLMError(f"Unexpected Ollama response structure: {exc}") from exc

#调用默认模型
def _call_dashscope(prompt: str, model: str) -> tuple[str, dict]:
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
        "temperature": settings.LLM_TEMPERATURE,
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

#统一的大模型调用入口：根据传入/默认的 provider 和 model 选择对应厂商的请求函数
def call_llm(prompt: str, provider: Optional[str] = None, model: Optional[str] = None) -> tuple[str, dict]:
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
