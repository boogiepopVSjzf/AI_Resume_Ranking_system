import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
env_path = BASE_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)

STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
TXT_DIR = STORAGE_DIR / "txts"
RESULTS_DIR = STORAGE_DIR / "results"

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MIN_UPLOAD_BYTES = 1 * 1024  # 1 KB — files smaller than this are considered invalid
MAX_FILENAME_LENGTH = 128
MIN_EXTRACTED_TEXT_CHARS = 30

# Primary LLM: Aliyun Dashscope (for feature_jzf compatibility)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Default routing to Dashscope
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "dashscope")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", LLM_MODEL)

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
