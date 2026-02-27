# tests/test_llm_service_fault_injection.py
# English comments as requested.

import json
import pytest

from services.llm_service import call_llm, LLMError


class FakeResponse:
    """A minimal fake response object to simulate requests.Response."""

    def __init__(self, status_code=200, json_data=None, text="", json_raises=False):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("Not JSON")
        return self._json_data


@pytest.fixture()
def patch_settings(monkeypatch):
    """Patch settings used by llm_service to ensure deterministic tests."""
    # Import inside fixture so it's the same module object used by llm_service.
    from config import settings

    monkeypatch.setattr(settings, "LLM_API_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_MODEL", "dummy-model")
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 5)
    return settings


def test_happy_path_chat_completions_message_content(monkeypatch, patch_settings):
    """Happy path: OpenAI-style chat/completions response."""
    import requests

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(
            status_code=200,
            json_data={"choices": [{"message": {"content": "OK"}}]},
            text="",
        )

    monkeypatch.setattr(requests, "post", fake_post)
    out = call_llm("hello")
    assert out == "OK"


def test_happy_path_top_level_text(monkeypatch, patch_settings):
    """Happy path: response returns top-level 'text' field."""
    import requests

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(status_code=200, json_data={"text": "OK"}, text="")

    monkeypatch.setattr(requests, "post", fake_post)
    out = call_llm("hello")
    assert out == "OK"


def test_edge_non_json_response_returns_text(monkeypatch, patch_settings):
    """Edge: resp.json() fails -> function should return resp.text."""
    import requests

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(status_code=200, json_data=None, text="plain text", json_raises=True)

    monkeypatch.setattr(requests, "post", fake_post)
    out = call_llm("hello")
    assert out == "plain text"


def test_negative_missing_api_url(monkeypatch):
    """Negative: missing LLM_API_URL should raise LLMError."""
    from config import settings

    monkeypatch.setattr(settings, "LLM_API_URL", "")
    with pytest.raises(LLMError) as ex:
        call_llm("hello")
    assert "未配置 LLM_API_URL" in str(ex.value)


def test_negative_requests_exception(monkeypatch, patch_settings):
    """Negative: requests.post raises exception -> LLMError('LLM 请求失败')."""
    import requests

    def fake_post(*args, **kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(LLMError) as ex:
        call_llm("hello")
    assert "LLM 请求失败" in str(ex.value)


def test_negative_http_500_with_detail(monkeypatch, patch_settings):
    """Negative: HTTP error >= 400 with detail should raise LLMError with detail."""
    import requests

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(status_code=500, json_data=None, text="server error", json_raises=True)

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(LLMError) as ex:
        call_llm("hello")
    assert "LLM 返回错误: 500" in str(ex.value)
    assert "server error" in str(ex.value)


def test_negative_http_401_no_detail(monkeypatch, patch_settings):
    """Negative: HTTP error >= 400 without detail should raise LLMError without detail."""
    import requests

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(status_code=401, json_data=None, text="   ", json_raises=True)

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(LLMError) as ex:
        call_llm("hello")
    assert "LLM 返回错误: 401" in str(ex.value)


def test_edge_unknown_json_structure_falls_back_to_json_dump(monkeypatch, patch_settings):
    """Edge: unknown dict structure should return json.dumps(data)."""
    import requests

    payload = {"foo": {"bar": [1, 2, 3]}, "answer": "maybe"}
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(status_code=200, json_data=payload, text="")

    monkeypatch.setattr(requests, "post", fake_post)

    out = call_llm("hello")
    # Ensure it returns a JSON string representation.
    loaded = json.loads(out)
    assert loaded["foo"]["bar"] == [1, 2, 3]
