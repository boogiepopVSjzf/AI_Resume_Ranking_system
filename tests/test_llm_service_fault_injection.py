"""
Fault Injection and Tolerance Tests for LLM Service.

This module uses the `responses` library to intercept HTTP calls made by `requests`.
It simulates various chaos engineering scenarios including infrastructure failures 
(Negative Paths) and LLM-specific hallucinations/mutations (Edge Cases) to ensure
the parsing pipeline does not crash under unexpected conditions.
"""

import json
import pytest
import requests
import responses

from services.llm_service import call_llm, LLMError

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture()
def patch_settings(monkeypatch):
    """
    Patch application settings to ensure deterministic tests without relying 
    on local environment variables.
    """
    from config import settings

    monkeypatch.setattr(settings, "LLM_API_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "LLM_API_KEY", "dummy-key")
    monkeypatch.setattr(settings, "LLM_MODEL", "dummy-model")
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 5)
    return settings

# ==============================================================================
# 1. Happy Path & Payload Verification
# ==============================================================================

@responses.activate
def test_happy_path_and_payload_verification(patch_settings):
    """
    Happy Path: Ensure the service returns the correct text and verify that 
    the exact expected payload and headers were sent to the external API.
    """
    mock_url = patch_settings.LLM_API_URL
    
    # Mock the successful HTTP 200 response
    responses.add(
        responses.POST,
        mock_url,
        json={"choices": [{"message": {"content": "Perfectly parsed JSON"}}]},
        status=200
    )

    prompt_text = "Extract the candidate name."
    output = call_llm(prompt_text)

    # Verify output
    assert output == "Perfectly parsed JSON"

    # Verify that exactly one HTTP call was made
    assert len(responses.calls) == 1
    
    # Verify the Request Payload (Crucial for ensuring the right model/prompt is used)
    request_body = json.loads(responses.calls[0].request.body)
    assert request_body["model"] == patch_settings.LLM_MODEL
    assert request_body["messages"][0]["content"] == prompt_text
    
    # Verify Authentication Headers
    request_headers = responses.calls[0].request.headers
    assert request_headers["Authorization"] == f"Bearer {patch_settings.LLM_API_KEY}"

# ==============================================================================
# 2. Edge Cases (LLM Hallucinations & Truncations)
# ==============================================================================

@responses.activate
def test_edge_case_markdown_wrapped_json(patch_settings):
    """
    Edge Case: The LLM acts chatty and wraps the requested JSON inside 
    Markdown code blocks. The service should handle or return the raw text 
    for the upstream cleaner to process.
    """
    chatty_content = "Here is the result:\n```json\n{\"name\": \"John\"}\n```"
    
    responses.add(
        responses.POST,
        patch_settings.LLM_API_URL,
        json={"choices": [{"message": {"content": chatty_content}}]},
        status=200
    )

    output = call_llm("Parse this")
    
    # If your call_llm has built-in regex cleaning, assert it equals {"name": "John"}.
    # Otherwise, assert it accurately returns the raw string without crashing.
    assert "```json" in output
    assert "John" in output

@responses.activate
def test_edge_case_max_tokens_truncated_response(patch_settings):
    """
    Edge Case: The LLM reaches the max token limit and returns an incomplete JSON.
    Even though the JSON is broken, the HTTP status is 200.
    """
    truncated_content = '{"name": "John", "skills": ["Python", "Java"'
    
    responses.add(
        responses.POST,
        patch_settings.LLM_API_URL,
        json={
            "choices": [{
                "message": {"content": truncated_content},
                "finish_reason": "length"  # Standard OpenAI truncation flag
            }]
        },
        status=200
    )

    output = call_llm("Parse this")
    
    # The system should not crash; it should return the truncated string.
    # The upstream schema validation (e.g., Pydantic) will catch the structural error.
    assert output == truncated_content

# ==============================================================================
# 3. Negative Paths (Infrastructure Failures)
# ==============================================================================

@responses.activate
def test_negative_path_rate_limiting_429(patch_settings):
    """
    Negative Path: The API provider limits the frequency of requests (HTTP 429).
    Should raise an LLMError explicitly mentioning the failure.
    """
    responses.add(
        responses.POST,
        patch_settings.LLM_API_URL,
        json={"error": {"message": "Rate limit exceeded"}},
        status=429
    )

    with pytest.raises(LLMError) as exc_info:
        call_llm("hello")
        
    assert "429" in str(exc_info.value)
    # The detail from the API should ideally be captured in the exception
    assert "Rate limit exceeded" in str(exc_info.value)

@responses.activate
def test_negative_path_network_timeout(patch_settings):
    """
    Negative Path: The external API takes too long to respond, causing a timeout.
    This prevents the main application thread from hanging indefinitely.
    """
    # Simulate a requests.exceptions.Timeout being raised during the POST call
    responses.add(
        responses.POST,
        patch_settings.LLM_API_URL,
        body=requests.exceptions.Timeout("Connection timed out")
    )

    with pytest.raises(LLMError) as exc_info:
        call_llm("hello")
        
    assert "API takes too long to respond" in str(exc_info.value)

@responses.activate
def test_negative_path_server_downtime_500(patch_settings):
    """
    Negative Path: The LLM provider's server crashes (HTTP 500).
    """
    responses.add(
        responses.POST,
        patch_settings.LLM_API_URL,
        body="Internal Server Error",
        status=500
    )

    with pytest.raises(LLMError) as exc_info:
        call_llm("hello")
        
    assert "500" in str(exc_info.value)

def test_negative_missing_api_url_config(monkeypatch):
    """
    Negative Path: Ensure the service fails fast if the environment 
    configuration is completely missing.
    """
    from config import settings
    monkeypatch.setattr(settings, "LLM_API_URL", "")
    
    with pytest.raises(LLMError) as exc_info:
        call_llm("hello")
        
    assert " configuration is completely missing" in str(exc_info.value)
