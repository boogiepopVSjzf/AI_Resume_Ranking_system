"""
Centralized constants for the application.
All magic numbers and repeated string literals should be defined here.
"""

# HTTP Status Codes
HTTP_400_BAD_REQUEST = 400
HTTP_413_PAYLOAD_TOO_LARGE = 413
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_502_BAD_GATEWAY = 502

# Error Messages
ERR_FILE_TOO_LARGE = "文件过大"
ERR_NO_FILE_SELECTED = "未选择文件"
ERR_FILENAME_TOO_LONG = "文件名过长"
ERR_FILE_EMPTY = "文件内容为空"
ERR_TEXT_EMPTY = "text 不能为空"
ERR_NO_FILENAME_PROVIDED = "未提供文件名"
ERR_FILE_CONTENT_EMPTY = "文件内容为空"
ERR_UNSUPPORTED_FILE_TYPE = "不支持的文件类型"

# File upload limits
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MIN_UPLOAD_BYTES = 1 * 1024  # 1 KB
MAX_FILENAME_LENGTH = 128
MIN_EXTRACTED_TEXT_CHARS = 30
MAX_BATCH_SIZE = 20  # Maximum files per batch upload

# PDF parsing thresholds
MULTICOLUMN_AVG_LINE_LEN = 40
MULTICOLUMN_MIN_LINES = 20

# Chunk size for file reading (1 MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024
