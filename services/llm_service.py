from __future__ import annotations

import certifi
import requests
from requests.adapters import HTTPAdapter
from typing import Optional

from config import settings
from utils.errors import LLMError


SUPPORTED_PROVIDERS = {"anthropic", "dashscope", "gemini", "openai", "ollama"}

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
    if provider == "anthropic":
        return settings.ANTHROPIC_MODEL
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


def _call_anthropic(prompt: str, model: str) -> tuple[str, dict]:
    if not settings.ANTHROPIC_API_KEY:
        raise LLMError("Missing ANTHROPIC_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        session = _build_https_session()
        response = session.post(
            settings.ANTHROPIC_API_URL,
            headers=headers,
            json=payload,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Anthropic request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"Anthropic API error: {response.status_code} - {response.text}")

    data = response.json()

    try:
        content_blocks = data["content"]
        text_chunks = [
            block["text"]
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "\n".join(chunk for chunk in text_chunks if chunk)
        usage = data.get("usage", {})
        if not content:
            raise KeyError("No text blocks in Anthropic response")
        return content, usage
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Unexpected Anthropic response structure: {exc}") from exc

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

# ─────────────────────────────────────────────────────────────
# Tool-calling (multi-turn agent) support
# ─────────────────────────────────────────────────────────────

def _normalize_tools_for_openai(tools: list[dict]) -> list[dict]:
    """Convert provider-agnostic tool defs (input_schema) to OpenAI/Dashscope format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def _normalize_tools_for_gemini(tools: list[dict]) -> list[dict]:
    """Convert to Gemini functionDeclarations format."""
    declarations = []
    for t in tools:
        declarations.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        })
    return [{"functionDeclarations": declarations}]


def _extract_tool_calls_openai(data: dict) -> list[dict]:
    tool_calls = []
    for tc in (data.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []):
        import json as _json
        args = tc.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                args = {}
        tool_calls.append({
            "id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "arguments": args,
        })
    return tool_calls


def _extract_tool_calls_anthropic(data: dict) -> list[dict]:
    tool_calls = []
    for block in data.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })
    return tool_calls


def _extract_tool_calls_gemini(data: dict) -> list[dict]:
    tool_calls = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": fc.get("name", ""),
                    "name": fc.get("name", ""),
                    "arguments": fc.get("args", {}),
                })
    return tool_calls


def _messages_to_gemini(messages: list[dict], system: Optional[str]) -> list[dict]:
    """Convert OpenAI-style messages to Gemini contents format."""
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": f"[System instructions]\n{system}"}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            gemini_role = "model"
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                parts = [{"functionCall": {"name": tc["name"], "args": tc["arguments"]}} for tc in tool_calls]
                contents.append({"role": gemini_role, "parts": parts})
                continue
        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {"name": msg.get("tool_name", "tool"), "response": {"content": content}}}],
            })
            continue
        else:
            gemini_role = "user"
        if isinstance(content, str):
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
    return contents


def _call_anthropic_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    system: Optional[str],
) -> tuple[Optional[str], list[dict], dict]:
    if not settings.ANTHROPIC_API_KEY:
        raise LLMError("Missing ANTHROPIC_API_KEY")

    anthropic_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }],
            })
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                blocks = []
                for tc in tool_calls:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                text = msg.get("content") or ""
                if text:
                    blocks.insert(0, {"type": "text", "text": text})
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                anthropic_messages.append({"role": "assistant", "content": content})
        else:
            anthropic_messages.append({"role": role, "content": content})

    payload: dict = {
        "model": model,
        "max_tokens": 4096,
        "temperature": settings.LLM_TEMPERATURE,
        "messages": anthropic_messages,
        "tools": tools,
    }
    if system:
        payload["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    try:
        sess = _build_https_session()
        response = sess.post(settings.ANTHROPIC_API_URL, headers=headers, json=payload, timeout=settings.LLM_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise LLMError(f"Anthropic tool-call request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"Anthropic API error: {response.status_code} - {response.text}")

    data = response.json()
    usage = data.get("usage", {})
    stop_reason = data.get("stop_reason", "")

    if stop_reason == "tool_use":
        return None, _extract_tool_calls_anthropic(data), usage

    text_chunks = [b["text"] for b in data.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(text_chunks).strip() or None, [], usage


def _call_openai_compatible_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    system: Optional[str],
    *,
    url: str,
    api_key: str,
) -> tuple[Optional[str], list[dict], dict]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            openai_messages.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": content,
            })
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                oai_tcs = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": __import__("json").dumps(tc["arguments"])}}
                    for tc in tool_calls
                ]
                entry: dict = {"role": "assistant", "tool_calls": oai_tcs}
                if content:
                    entry["content"] = content
                openai_messages.append(entry)
            else:
                openai_messages.append({"role": "assistant", "content": content})
        else:
            openai_messages.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": openai_messages,
        "tools": _normalize_tools_for_openai(tools),
        "temperature": settings.LLM_TEMPERATURE,
    }

    try:
        sess = _build_https_session()
        response = sess.post(url, headers=headers, json=payload, timeout=settings.LLM_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise LLMError(f"Tool-call request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"API error {response.status_code}: {response.text}")

    data = response.json()
    usage = data.get("usage", {})
    choice = (data.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason", "")

    if finish_reason == "tool_calls":
        return None, _extract_tool_calls_openai(data), usage

    content = choice.get("message", {}).get("content") or ""
    return content or None, [], usage


def _call_gemini_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    system: Optional[str],
) -> tuple[Optional[str], list[dict], dict]:
    if not settings.GEMINI_API_KEY:
        raise LLMError("Missing GEMINI_API_KEY")

    url = f"{settings.GEMINI_API_URL_TEMPLATE.format(model=model)}?key={settings.GEMINI_API_KEY}"
    contents = _messages_to_gemini(messages, system)
    payload = {
        "contents": contents,
        "tools": _normalize_tools_for_gemini(tools),
        "generationConfig": {"temperature": settings.LLM_TEMPERATURE},
    }

    try:
        sess = _build_https_session()
        response = sess.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=settings.LLM_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise LLMError(f"Gemini tool-call request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"Gemini API error: {response.status_code} - {response.text}")

    data = response.json()
    usage = data.get("usageMetadata", {})
    tool_calls = _extract_tool_calls_gemini(data)
    if tool_calls:
        return None, tool_calls, usage

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text, [], usage
    except (KeyError, IndexError, TypeError):
        return None, [], usage


def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
) -> tuple[Optional[str], list[dict], dict]:
    """
    Multi-turn tool-calling LLM call.

    Returns (text_response | None, tool_calls, usage).
    Exactly one of text_response or tool_calls will be non-empty per call.
    """
    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model(resolved_provider, model)

    if resolved_provider == "anthropic":
        return _call_anthropic_with_tools(messages, tools, resolved_model, system)

    if resolved_provider in ("openai",):
        if not settings.OPENAI_API_KEY:
            raise LLMError("Missing OPENAI_API_KEY")
        return _call_openai_compatible_with_tools(
            messages, tools, resolved_model, system,
            url=settings.OPENAI_API_URL,
            api_key=settings.OPENAI_API_KEY,
        )

    if resolved_provider == "dashscope":
        if not settings.LLM_API_KEY:
            raise LLMError("Missing LLM_API_KEY")
        return _call_openai_compatible_with_tools(
            messages, tools, resolved_model, system,
            url=f"{settings.LLM_BASE_URL}/chat/completions",
            api_key=settings.LLM_API_KEY,
        )

    if resolved_provider == "gemini":
        return _call_gemini_with_tools(messages, tools, resolved_model, system)

    if resolved_provider == "ollama":
        if not settings.OLLAMA_TOOL_CALLING:
            raise LLMError(
                "Ollama tool-calling is disabled. Set OLLAMA_TOOL_CALLING=true to enable "
                "(requires a model that supports function calling)."
            )
        ollama_chat_url = settings.OLLAMA_API_URL.replace("/api/generate", "/api/chat")
        return _call_openai_compatible_with_tools(
            messages, tools, resolved_model, system,
            url=ollama_chat_url,
            api_key="ollama",
        )

    raise LLMError(f"Unsupported provider for tool-calling: {resolved_provider}")


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
    if resolved_provider == "anthropic":
        return _call_anthropic(prompt, resolved_model)
    if resolved_provider == "ollama":
        return _call_ollama(prompt, resolved_model)

    raise LLMError(f"Unsupported provider: {resolved_provider}")
