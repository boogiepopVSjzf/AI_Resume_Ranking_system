import os
from pathlib import Path

from dotenv import load_dotenv

from utils.constants import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_BYTES,
    MIN_UPLOAD_BYTES,
    MAX_FILENAME_LENGTH,
    MIN_EXTRACTED_TEXT_CHARS,
)

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

# Primary LLM: Aliyun Dashscope (for feature_jzf compatibility)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-max")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Default routing to Dashscope
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "dashscope")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", LLM_MODEL)

# Role-specific LLM routing.
# Parse and query rewrite share one LLM configuration because both tasks are
# structured extraction/rewrite tasks that benefit from consistent JSON output.
PARSE_QUERY_LLM_PROVIDER = os.getenv("PARSE_QUERY_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip()
PARSE_QUERY_LLM_MODEL = os.getenv("PARSE_QUERY_LLM_MODEL", "").strip() or None
SCHEMA_LLM_PROVIDER = os.getenv("SCHEMA_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip()
SCHEMA_LLM_MODEL = os.getenv("SCHEMA_LLM_MODEL", "").strip() or None
SCORING_LLM_PROVIDER = os.getenv("SCORING_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip()
SCORING_LLM_MODEL = os.getenv("SCORING_LLM_MODEL", "").strip() or None
SCHEMA_CACHE_ENABLED = _env_bool("SCHEMA_CACHE_ENABLED", True)
SCHEMA_CACHE_MAX_SIZE = _env_int("SCHEMA_CACHE_MAX_SIZE", 128)
SCHEMA_PROMPT_VERSION = os.getenv("SCHEMA_PROMPT_VERSION", "2026-04-24-v1")
QUERY_REWRITE_CACHE_ENABLED = _env_bool("QUERY_REWRITE_CACHE_ENABLED", True)
QUERY_REWRITE_CACHE_MAX_SIZE = _env_int("QUERY_REWRITE_CACHE_MAX_SIZE", 256)
SCORING_CACHE_ENABLED = _env_bool("SCORING_CACHE_ENABLED", True)
SCORING_CACHE_MAX_SIZE = _env_int("SCORING_CACHE_MAX_SIZE", 1024)
SCORING_PROMPT_VERSION = os.getenv("SCORING_PROMPT_VERSION", "2026-04-24-v1")

# Alternative providers (Gemini, OpenAI, Ollama)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

# Embedding (sentence-transformers)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
PRELOAD_EMBEDDING_MODEL = _env_bool("PRELOAD_EMBEDDING_MODEL", False)
INCLUDE_EMBEDDING_IN_RESPONSE = _env_bool("INCLUDE_EMBEDDING_IN_RESPONSE", False)

# Database persistence
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DATABASE_SSLMODE = os.getenv("DATABASE_SSLMODE", "require").strip() or "require"
ENABLE_DB_PERSISTENCE = _env_bool("ENABLE_DB_PERSISTENCE", bool(DATABASE_URL))
DB_AUTO_INIT = _env_bool("DB_AUTO_INIT", True)

# AWS S3
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "").strip()
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip() or None
ENABLE_S3_STORAGE = _env_bool(
    "ENABLE_S3_STORAGE",
    bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_REGION and S3_BUCKET_NAME),
)
