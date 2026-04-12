import json
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
from services.job_query_rewrite_service import embed_search_query
from services.resume_storage_bundle import build_resume_storage_bundle
from storage.postgres_store import save_resume_bundle
from storage.s3_storage import upload_resume_source_file

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
    DatabaseError,
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
    DatabaseError: HTTP_502_BAD_GATEWAY,
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


def _parse_single_resume_file(
    *,
    filename: str,
    ext: str,
    content: bytes,
) -> dict:
    """Parse one resume file end-to-end and return the API payload."""
    start_time = time.time()

    result = process_upload(ext, content)
    structured, usage = _extract_structured(result.text, result.resume_id)

    bundle = build_resume_storage_bundle(structured)
    pdf_upload = None
    if settings.ENABLE_S3_STORAGE:
        pdf_upload = upload_resume_source_file(
            resume_id=result.resume_id,
            filename=filename or f"{result.resume_id}{ext}",
            ext=ext,
            content=content,
        )

    persisted_to_db = _persist_bundle_if_enabled(
        resume_id=result.resume_id,
        bundle=bundle,
        source_file_name=filename,
        source_file_type=ext,
        pdf_storage_bucket=pdf_upload.bucket if pdf_upload else None,
        pdf_storage_key=pdf_upload.key if pdf_upload else None,
        pdf_mime_type=pdf_upload.mime_type if pdf_upload else None,
    )

    duration = time.time() - start_time
    logger.info("Parsed resume %s in %.2f seconds", result.resume_id, duration)
    return {
        "resume_id": result.resume_id,
        "resume": bundle,
        "persisted_to_db": persisted_to_db,
        "pdf_uploaded_to_storage": pdf_upload is not None,
        "pdf_storage_bucket": pdf_upload.bucket if pdf_upload else None,
        "pdf_storage_key": pdf_upload.key if pdf_upload else None,
        "usage": usage,
        "duration_seconds": round(duration, 2),
    }


def _persist_bundle_if_enabled(
    *,
    resume_id: Optional[str],
    bundle: dict,
    source_file_name: Optional[str] = None,
    source_file_type: Optional[str] = None,
    pdf_storage_bucket: Optional[str] = None,
    pdf_storage_key: Optional[str] = None,
    pdf_mime_type: Optional[str] = None,
) -> bool:
    """Persist a parsed resume bundle to Postgres when configured."""
    if not resume_id or not settings.ENABLE_DB_PERSISTENCE:
        return False

    save_resume_bundle(
        resume_id=resume_id,
        bundle=bundle,
        source_file_name=source_file_name,
        source_file_type=source_file_type,
        pdf_storage_bucket=pdf_storage_bucket,
        pdf_storage_key=pdf_storage_key,
        pdf_mime_type=pdf_mime_type,
    )
    return True


@router.get("/")
def index():
    """Health check and API navigation."""
    return JSONResponse({
        "message": "ok",
        "docs": "/docs",
        "endpoints": [
            "/api/upload",
            "/api/extract",
            "/api/parse",
            "/api/parse/batch",
            "/api/job-context",
            "/api/query-rewrite",
        ],
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
    try:
        ext = validate_filename(file.filename)
        content = await _read_upload_content(file, _get_content_length(request))
    except HTTPException:
        raise
    except Exception as exc:
        _raise_http_exception(exc)

    try:
        payload = _parse_single_resume_file(
            filename=file.filename or "",
            ext=ext,
            content=content,
        )
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse(payload)


@router.post("/api/parse/batch")
async def parse_resume_batch(request: Request, files: list[UploadFile] = File(...)):
    """Batch parse multiple resumes into structured data, embeddings, and DB records."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"批量解析最多支持 {MAX_BATCH_SIZE} 个文件，当前 {len(files)} 个",
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
            logger.error("[PARSE_BATCH] Skipping %s: %s", filename, exc.detail)
            failed.append({"filename": filename, "reason": exc.detail})
            continue

        try:
            payload = _parse_single_resume_file(
                filename=filename,
                ext=ext,
                content=content,
            )
            payload["filename"] = filename
            succeeded.append(payload)
        except Exception as exc:
            logger.error("[PARSE_BATCH] %s failed: %s", filename, exc)
            failed.append({"filename": filename, "reason": str(exc)})

    return JSONResponse({
        "total": len(files),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "succeeded": succeeded,
        "failed": failed,
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

    bundle = build_resume_storage_bundle(structured)
    persisted_to_db = _persist_bundle_if_enabled(
        resume_id=resume_id,
        bundle=bundle,
    )

    return JSONResponse({
        "resume": bundle,
        "persisted_to_db": persisted_to_db,
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
            result["search_query_embedding"] = embed_search_query(query)
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
        "search_query_embedding": embed_search_query(query),
        "usage": usage,
    })
