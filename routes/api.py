import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from services.extract_service import extract_structured_resume
from services.upload_service import (
    process_single_file_in_batch,
    process_upload,
    validate_batch_file,
    validate_filename,
)
from services.job_context_service import (
    JDSourceConflict,
    JobContextEmpty,
    build_job_context,
    resolve_jd_body,
)
from services.job_query_rewrite_service import rewrite_merged_context
from services.resume_storage_bundle import build_resume_storage_bundle
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
    CorruptedPDFError,
    DocumentExtractError,
    EncryptedPDFError,
    FileSizeError,
    InvalidFileType,
    InvalidResumeError,
    LLMError,
    LLMParseError,
)
from utils.logger import get_logger
from schemas.models import ExtractionInput

router = APIRouter()
logger = get_logger("api")

# Exception to HTTP status code mapping
_EXCEPTION_STATUS_MAP = {
    InvalidFileType: HTTP_400_BAD_REQUEST,
    FileSizeError: HTTP_400_BAD_REQUEST,
    InvalidResumeError: HTTP_422_UNPROCESSABLE_ENTITY,
    LLMParseError: HTTP_422_UNPROCESSABLE_ENTITY,
    EncryptedPDFError: HTTP_422_UNPROCESSABLE_ENTITY,
    CorruptedPDFError: HTTP_422_UNPROCESSABLE_ENTITY,
    DocumentExtractError: HTTP_422_UNPROCESSABLE_ENTITY,
    LLMError: HTTP_502_BAD_GATEWAY,
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
        "endpoints": ["/api/upload", "/api/extract", "/api/parse", "/api/job-context", "/api/query-rewrite"],
    })


@router.post("/api/upload")
async def upload_resume(request: Request, files: list[UploadFile] = File(...)):
    """Upload and convert one or more resume files to text."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"单次上传最多支持 {MAX_BATCH_SIZE} 个文件，当前 {len(files)} 个",
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
            logger.error("Skipping %s: %s", filename, exc.detail)
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
    start_time = time.time()
    try:
        ext = validate_filename(file.filename)
        content = await _read_upload_content(file, _get_content_length(request))
        result = process_upload(ext, content)
        structured, usage = _extract_structured(result.text, result.resume_id)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_http_exception(exc)

    duration = time.time() - start_time
    json_text = structured.model_dump_json(ensure_ascii=False)
    save_result_json(result.resume_id, json_text)

    bundle = build_resume_storage_bundle(structured)
    logger.info("Parsed resume %s in %.2f seconds", result.resume_id, duration)

    return JSONResponse({
        "resume_id": result.resume_id,
        "resume": bundle,
        "usage": usage,
        "duration_seconds": round(duration, 2),
    })


@router.post("/api/extract")
async def extract_resume(payload: dict):
    """Extract structured data from resume text."""
    text = payload.get("text")
    resume_id = payload.get("resume_id") if isinstance(payload.get("resume_id"), str) else None

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="text 不能为空")

    try:
        structured, usage = _extract_structured(text, resume_id)
    except Exception as exc:
        _raise_http_exception(exc)

    json_text = structured.model_dump_json(ensure_ascii=False)
    if resume_id:
        save_result_json(resume_id, json_text)

    bundle = build_resume_storage_bundle(structured)

    return JSONResponse({
        "resume": bundle,
        "usage": usage,
    })


@router.post("/api/job-context")
async def submit_job_context(
    hr_note: str = Form(""),
    jd_text: str = Form(""),
    jd_file: Optional[UploadFile] = File(None),
    rewrite: bool = Form(False),
):
    """Accept HR note and/or JD (text or PDF) and return merged context.

    When ``rewrite=True``, the merged context is additionally sent through
    the LLM query-rewrite pipeline and the response includes
    ``standardized_query`` and ``query_rewrite_usage``.
    """
    jd_file_content: Optional[bytes] = None
    if jd_file is not None:
        raw = await jd_file.read()
        if raw:
            jd_file_content = raw

    try:
        jd_body = resolve_jd_body(jd_text, jd_file_content)
        result = build_job_context(hr_note, jd_body)
    except JDSourceConflict as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobContextEmpty as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    if rewrite:
        try:
            query, rewrite_usage = rewrite_merged_context(result["merged_context"])
            result["standardized_query"] = query.model_dump()
            result["query_rewrite_usage"] = rewrite_usage
        except Exception as exc:
            _raise_http_exception(exc)

    return JSONResponse(result)


@router.post("/api/query-rewrite")
async def query_rewrite(payload: dict):
    """Rewrite merged_context into hard_filters + search_query via LLM."""
    merged_context = payload.get("merged_context")
    if not isinstance(merged_context, str) or not merged_context.strip():
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="merged_context is required and must be a non-empty string",
        )

    try:
        query, usage = rewrite_merged_context(merged_context)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse({
        "hard_filters": query.hard_filters.model_dump(),
        "search_query": query.search_query,
        "usage": usage,
    })
