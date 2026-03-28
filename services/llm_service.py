from __future__ import annotations

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
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = requests.packages.urllib3.util.ssl_.create_urllib3_context(ciphers=CIPHERS)
        kwargs["ssl_context"] = context
        return super().proxy_manager_for(*args, **kwargs)

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
def _call_gemini(prompt: str, model: str) -> str:
    if not settings.GEMINI_API_KEY:
        raise LLMError("Missing GEMINI_API_KEY")
#没apikey直接报错
    url = settings.GEMINI_API_URL_TEMPLATE.format(model=model)
    url = f"{url}?key={settings.GEMINI_API_KEY}"
#把apikey作为query参数传递到url中
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1, #控制生成文本的随机性，值越小越确定，值越大越随机
            "maxOutputTokens": 2048, #最大输出token数
            "responseMimeType": "application/json", #响应类型，这里指定为json
        },
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS, #设置超时时间
        )
    except requests.RequestException as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc
#捕获网络层问题（超时、连接失败、DNS 等），统一包装成 LLMError 往上抛。
    if response.status_code != 200:
        raise LLMError(f"Gemini API error: {response.status_code} - {response.text}")

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Gemini response structure: {exc}") from exc

#调用 OpenAI 模型
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
        session = requests.Session()
        session.mount("https://", SSLAdapter())
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected OpenAI response structure: {exc}") from exc

#调用 Ollama 模型
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

#调用默认模型
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
        session = requests.Session()
        session.mount("https://", SSLAdapter())
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
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected Dashscope response structure: {exc}") from exc

#统一的大模型调用入口：根据传入/默认的 provider 和 model 选择对应厂商的请求函数
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