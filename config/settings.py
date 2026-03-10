import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
env_path = BASE_DIR / ".env"

if env_path.exists():import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
env_path = BASE_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)

STORAGE_DIR = BASE_DIR / "storage"
PDF_DIR = STORAGE_DIR / "pdfs"
TXT_DIR = STORAGE_DIR / "txts"
RESULTS_DIR = STORAGE_DIR / "results"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_MB = 20
LLM_TIMEOUT_SECONDS = 180

# Default routing
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")

# Gemini config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenAI-compatible config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

# Ollama config
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",  "deepseek-r1:1.5b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_MB = 20
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "") 
LLM_TIMEOUT_SECONDS = 60

LLM_MODEL = "gemini-3-flash-preview"
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
