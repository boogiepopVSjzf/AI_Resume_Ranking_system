import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
env_path = BASE_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f" Successfully loaded .env from {env_path}")
else:
    print(f"Error: .env file NOT FOUND at {env_path}")

STORAGE_DIR = BASE_DIR / "storage"
PDF_DIR = STORAGE_DIR / "pdfs"
TXT_DIR = STORAGE_DIR / "txts"
RESULTS_DIR = STORAGE_DIR / "results"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_MB = 20
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "") 
LLM_TIMEOUT_SECONDS = 60

LLM_MODEL = "gemini-3-flash-preview"
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
