import json
from typing import Any, Dict

import requests

from config import settings
from utils.errors import AppError,LLMError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Define custom network exceptions
class TransientNetworkError(Exception):
    pass

def is_transient_error(exception):
    """
    Return True if the error is worth retrying.
    Do NOT retry on 400 (Bad Request) or 401 (Unauthorized).
    """
    return isinstance(exception, TransientNetworkError)

# Retry up to 4 times, waiting 2^x * 1 seconds between each retry (2s, 4s, 8s)
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientNetworkError)
)
def call_llm_with_retry(prompt: str) -> str:
    """
    Makes the actual HTTP call to the LLM. 
    Handles retries for transient errors automatically.
    """
    try:
        response = requests.post(
            "YOUR_API_URL", 
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=10
        )
        
        # If API returns 429 (Rate Limit) or 500+ (Server Error), raise custom exception to trigger retry
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientNetworkError(f"Temporary API failure: {response.status_code}")
            
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        # Timeouts are also transient, trigger retry
        raise TransientNetworkError("Request timed out")
    except requests.exceptions.RequestException as e:
        # For non-transient errors (like 401 Unauthorized), wrap and raise immediately
        raise LLMError(f"Permanent LLM failure: {str(e)}")

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
