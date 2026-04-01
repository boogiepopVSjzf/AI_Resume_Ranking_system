import time
import json
import logging
from functools import wraps
from datetime import datetime
import requests

# Set up a dedicated logger for LLM telemetry
# In a real production system, this might pipe to ELK, Datadog, or Sentry
llm_error_logger = logging.getLogger("LLM_Telemetry")
llm_error_logger.setLevel(logging.ERROR)

# Optionally, configure a dedicated file handler for LLM errors
# so they don't get lost in standard application logs.
file_handler = logging.FileHandler("storage/results/llm_production_errors.jsonl")
llm_error_logger.addHandler(file_handler)

def with_llm_error_tracking(func):
    """
    A decorator to monitor, catch, and structure log errors 
    that occur during real LLM API calls in the production environment.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract prompt from kwargs or args for context logging (truncated for safety)
        prompt_context = kwargs.get("prompt", args[0] if args else "Unknown Prompt")
        safe_prompt_snippet = str(prompt_context)[:100] + "..." 

        try:
            # Execute the actual LLM call
            result = func(*args, **kwargs)
            return result

        except requests.exceptions.Timeout as e:
            # Catch strict network timeouts
            latency = time.time() - start_time
            _log_llm_failure("TIMEOUT", str(e), latency, safe_prompt_snippet)
            raise 

        except requests.exceptions.HTTPError as e:
            # Catch HTTP level errors (e.g., 429 Too Many Requests, 500 Server Error)
            latency = time.time() - start_time
            status_code = e.response.status_code if e.response else "Unknown"
            error_category = f"HTTP_{status_code}"
            
            _log_llm_failure(error_category, e.response.text if e.response else str(e), latency, safe_prompt_snippet)
            raise

        except requests.exceptions.RequestException as e:
            # Catch DNS resolution failures, connection resets, etc.
            latency = time.time() - start_time
            _log_llm_failure("CONNECTION_ERROR", str(e), latency, safe_prompt_snippet)
            raise

        except Exception as e:
            # Catch unexpected Python runtime errors (e.g., JSON parsing logic inside the service)
            latency = time.time() - start_time
            _log_llm_failure("UNEXPECTED_RUNTIME_ERROR", str(e), latency, safe_prompt_snippet)
            raise

    return wrapper

def _log_llm_failure(error_type: str, details: str, latency: float, prompt_snippet: str):
    """
    Helper function to format the error into a structured JSON log.
    Structured logs are much easier to query in tools like Kibana/Splunk.
    """
    error_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "LLM_Service",
        "error_type": error_type,
        "latency_seconds": round(latency, 2),
        "details": details,
        "prompt_snippet": prompt_snippet
    }
    
    # Log as a JSON string
    llm_error_logger.error(json.dumps(error_payload))
