#python -m pytest -q tests/test_llm_monitor.py

import json
import logging
import requests
import pytest

from llm_monitor import with_llm_error_tracking, llm_error_logger


@pytest.fixture()
def redirect_llm_monitor_log(tmp_path, monkeypatch):
    """
    Redirect the FileHandler to a temp file for test isolation.
    """
    log_file = tmp_path / "llm_production_errors.jsonl"

    # Remove existing handlers to avoid writing to real file
    for h in list(llm_error_logger.handlers):
        llm_error_logger.removeHandler(h)

    handler = logging.FileHandler(log_file)
    llm_error_logger.addHandler(handler)
    llm_error_logger.setLevel(logging.ERROR)

    return log_file


def read_last_jsonl_line(path):
    """
    Read the last line from a jsonl file and parse it as JSON.
    """
    lines = path.read_text().splitlines()
    assert lines, "No logs written"
    return json.loads(lines[-1])


def test_monitor_logs_timeout(redirect_llm_monitor_log):
    @with_llm_error_tracking
    def fake_llm_call(prompt: str):
        raise requests.exceptions.Timeout("Connection timed out")

    with pytest.raises(requests.exceptions.Timeout):
        fake_llm_call(prompt="hello")

    payload = read_last_jsonl_line(redirect_llm_monitor_log)
    assert payload["error_type"] == "TIMEOUT"
    assert "timed out" in payload["details"]
    assert payload["component"] == "LLM_Service"
    assert "hello" in payload["prompt_snippet"]


def test_monitor_logs_unexpected_runtime_error(redirect_llm_monitor_log):
    @with_llm_error_tracking
    def fake_llm_call(prompt: str):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fake_llm_call(prompt="hello")

    payload = read_last_jsonl_line(redirect_llm_monitor_log)
    assert payload["error_type"] == "UNEXPECTED_RUNTIME_ERROR"
    assert "boom" in payload["details"]
