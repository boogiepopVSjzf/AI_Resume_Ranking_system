import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

STORAGE_DIR = BASE_DIR / "storage"
PDF_DIR = STORAGE_DIR / "pdfs"
TXT_DIR = STORAGE_DIR / "txts"
RESULTS_DIR = STORAGE_DIR / "results"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MIN_UPLOAD_BYTES = 1 * 1024  # 1 KB — files smaller than this are considered invalid

LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen3.5-122b-a10b"
LLM_TIMEOUT_SECONDS = 60

LLM_API_KEY = os.getenv("LLM_API_KEY")
