import json
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from services.extract_service import extract_structured_resume
from services.upload_service import (
    process_single_file_in_batch,
    process_upload,
    validate_batch_file,
    validate_filename,
)
from storage.file_store import save_result_json

from utils.constants import (
    DEFAULT_CHUNK_SIZE,
    ERR_FILE_CONTENT_EMPTY,
    ERR_FILE_EMPTY,
    ERR_FILE_TOO_LARGE,
    HTTP_400_BAD_REQUEST,
    HTTP_413_PAYLOAD_TOO_LARGE,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_502_BAD_GATEWAY,
    MAX_BATCH_SIZE,
)
from utils.errors import (
    AppError,
    CorruptedPDFError,
    DocumentExtractError,
    EncryptedPDFError,
    FileSizeError,
    InvalidFileType,
    LLMParseError,
    NotResumeError,
)
from utils.logger import get_logger
from schemas.models import ExtractionInput

router = APIRouter()
logger = get_logger("api")

# Exception to HTTP status code mapping
_EXCEPTION_STATUS_MAP = {
    InvalidFileType: HTTP_400_BAD_REQUEST,
    FileSizeError: HTTP_400_BAD_REQUEST,
    EncryptedPDFError: HTTP_422_UNPROCESSABLE_ENTITY,
    CorruptedPDFError: HTTP_422_UNPROCESSABLE_ENTITY,
    DocumentExtractError: HTTP_422_UNPROCESSABLE_ENTITY,
    NotResumeError: HTTP_422_UNPROCESSABLE_ENTITY,
    LLMParseError: HTTP_502_BAD_GATEWAY,
    AppError: HTTP_502_BAD_GATEWAY,
}


def _raise_http_exception(exc: Exception) -> None:
    """Convert application exception to HTTPException."""
    for exc_type, status_code in _EXCEPTION_STATUS_MAP.items():
        if isinstance(exc, exc_type):
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


async def _read_upload_content(file: UploadFile, content_length: Optional[int]) -> bytes:
    """Read and validate upload file content."""
    if content_length is not None and content_length > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=HTTP_413_PAYLOAD_TOO_LARGE, detail=ERR_FILE_TOO_LARGE)

    data = bytearray()
    total = 0
    while True:
        chunk = await file.read(DEFAULT_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=HTTP_413_PAYLOAD_TOO_LARGE, detail=ERR_FILE_TOO_LARGE)
        data.extend(chunk)
    
    if not data:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=ERR_FILE_EMPTY)
    return bytes(data)


def _get_content_length(request: Request) -> Optional[int]:
    """Extract content-length header as int, or None if invalid."""
    val = request.headers.get("content-length")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def _extract_structured(text: str, resume_id: Optional[str]):
    """Call LLM extract service."""
    return extract_structured_resume(ExtractionInput(text=text, resume_id=resume_id))


@router.get("/")
def index():
    """Health check and API navigation."""
    return JSONResponse({
        "message": "ok",
        "docs": "/docs",
        "endpoints": ["/api/upload", "/api/extract", "/api/parse"],
    })


@router.post("/api/upload")
async def upload_resume(request: Request, file: UploadFile = File(...)):
    """Upload and convert a resume file to text."""
    try:
        ext = validate_filename(file.filename)
        content = await _read_upload_content(file, _get_content_length(request))
        result = process_upload(ext, content)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_http_exception(exc)

    logger.info("Uploaded resume %s", result.resume_id)
    return JSONResponse({
        "resume_id": result.resume_id,
        "text": result.text,
        "txt_path": result.txt_path,
    })


@router.post("/api/upload/batch")
async def upload_resume_batch(request: Request, files: list[UploadFile] = File(...)):
    """Batch upload and convert multiple resume files."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"批量上传最多支持 {MAX_BATCH_SIZE} 个文件，当前 {len(files)} 个"
        )

    succeeded, failed = [], []

    for file in files:
        ext, filename, failure = validate_batch_file(file.filename)
        if failure:
            failed.append(failure)
            continue

        try:
            content = await _read_upload_content(file, None)
        except HTTPException as exc:
            logger.error("[BATCH] Skipping %s: %s", filename, exc.detail)
            failed.append({"filename": filename, "reason": exc.detail})
            continue

        success, failure = process_single_file_in_batch(filename, ext, content)
        if success:
            succeeded.append(success)
        if failure:
            failed.append(failure)

    return JSONResponse({
        "total": len(files),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "succeeded": succeeded,
        "failed": failed,
    })


@router.post("/api/parse")
async def parse_resume(request: Request, file: UploadFile = File(...)):
    """Upload, convert to text, and extract structured data from a resume."""
    try:
        ext = validate_filename(file.filename)
        content = await _read_upload_content(file, _get_content_length(request))
        result = process_upload(ext, content)
        structured = _extract_structured(result.text, result.resume_id)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_http_exception(exc)

    json_text = structured.model_dump_json(ensure_ascii=False)
    save_result_json(result.resume_id, json_text)

    

    logger.info("Parsed resume %s", result.resume_id)

    return JSONResponse({
        "resume_id": result.resume_id,
        "result": json.loads(json_text),
    })


@router.post("/api/extract")
async def extract_resume(payload: dict):
    """Extract structured data from resume text."""
    text = payload.get("text")
    resume_id = payload.get("resume_id") if isinstance(payload.get("resume_id"), str) else None

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="text 不能为空")

    try:
        structured = _extract_structured(text, resume_id)
    except Exception as exc:
        _raise_http_exception(exc)

    json_text = structured.model_dump_json(ensure_ascii=False)
    if resume_id:
        save_result_json(resume_id, json_text)
        

    return JSONResponse(json.loads(json_text))
