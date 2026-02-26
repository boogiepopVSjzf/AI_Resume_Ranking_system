import json
from typing import Any, Dict

import requests

from config import settings
from utils.errors import AppError


class LLMError(AppError):
    pass


def _normalize_url(raw_url: str) -> str:
    url = raw_url.rstrip("/")
    if url.endswith("/v1") or url.endswith("/compatible-mode/v1"):
        return f"{url}/chat/completions"
    return url


def _is_chat_endpoint(url: str) -> bool:
    lowered = url.lower()
    return "chat/completions" in lowered or "compatible-mode" in lowered


def _build_payload(prompt: str, url: str) -> Dict[str, Any]:
    if _is_chat_endpoint(url):
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
        }
        if settings.LLM_MODEL:
            payload["model"] = settings.LLM_MODEL
        return payload
    if settings.LLM_MODEL:
        return {"model": settings.LLM_MODEL, "prompt": prompt}
    return {"prompt": prompt}


def call_llm(prompt: str) -> str:
    if not settings.LLM_API_URL:
        raise LLMError("未配置 LLM_API_URL")

    url = _normalize_url(settings.LLM_API_URL)
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=_build_payload(prompt, url),
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise LLMError("LLM 请求失败") from exc

    if resp.status_code >= 400:
        detail = resp.text.strip()
        if detail:
            raise LLMError(f"LLM 返回错误: {resp.status_code} {detail}")
        raise LLMError(f"LLM 返回错误: {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        return resp.text

    if isinstance(data, dict):
        for key in ("text", "content", "output"):
            if key in data and isinstance(data[key], str):
                return data[key]
        for key in ("choices",):
            if key in data and isinstance(data[key], list) and data[key]:
                first = data[key][0]
                if isinstance(first, dict):
                    if "text" in first and isinstance(first["text"], str):
                        return first["text"]
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    delta = first.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        return delta["content"]

    return json.dumps(data, ensure_ascii=False)
