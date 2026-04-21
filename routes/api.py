import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from services.extract_service import extract_structured_resume
from services.upload_service import (
    process_upload_with_resume_id,
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
from services.scoring_schema_service import build_scoring_schema_payload
from services.scoring_service import score_resume_with_schema
from storage.postgres_store import (
    find_best_scoring_schema,
    get_feedback_examples,
    get_resumes_by_ids,
    query_resume_ids_by_hard_filters,
    query_similar_resumes,
    save_scoring_feedback,
    save_scoring_schema,
    save_resume_bundle,
)
from storage.s3_storage import upload_resume_source_file

from utils.constants import (
    DEFAULT_CHUNK_SIZE,
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
from schemas.job_query import (
    HardFilters,
    VectorRetrieveRequest,
    VectorRetrieveResponse,
)
from schemas.models import ExtractionInput
from fastapi import Depends
from auth.deps import require_hr_or_internal, require_internal

router = APIRouter()
logger = get_logger("api")
_S3_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4)

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


def _bundle_for_response(bundle: dict) -> dict:
    """Trim large internal fields that do not need to travel back to the client."""
    if settings.INCLUDE_EMBEDDING_IN_RESPONSE:
        return bundle
    response_bundle = dict(bundle)
    response_bundle.pop("embedding", None)
    return response_bundle


def _parse_single_resume_file(
    *,
    filename: str,
    ext: str,
    content: bytes,
) -> dict:
    """Parse one resume file end-to-end and return the API payload."""
    start_time = time.time()
    s3_upload_future = None
    resume_id = uuid4().hex

    if settings.ENABLE_S3_STORAGE:
        s3_upload_future = _S3_UPLOAD_EXECUTOR.submit(
            upload_resume_source_file,
            resume_id=resume_id,
            filename=filename or f"resume{ext}",
            ext=ext,
            content=content,
        )

    result = process_upload_with_resume_id(ext, content, resume_id)

    structured, usage = _extract_structured(result.text, result.resume_id)

    bundle = build_resume_storage_bundle(structured)
    pdf_upload = None
    if s3_upload_future is not None:
        pdf_upload = s3_upload_future.result()

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
        "resume": _bundle_for_response(bundle),
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


def _parse_resume_ids(raw_resume_ids: str) -> list[str]:
    """Accept JSON array, comma-separated, or newline-separated resume IDs."""
    if not raw_resume_ids or not raw_resume_ids.strip():
        raise ValueError("resume_ids cannot be empty")

    raw_resume_ids = raw_resume_ids.strip()
    if raw_resume_ids.startswith("["):
        loaded = json.loads(raw_resume_ids)
        if not isinstance(loaded, list):
            raise ValueError("resume_ids JSON must be an array")
        raw_items = [str(item).strip() for item in loaded]
    else:
        raw_items = [
            item.strip()
            for line in raw_resume_ids.splitlines()
            for item in line.split(",")
        ]

    resume_ids = []
    seen = set()
    for resume_id in raw_items:
        if resume_id and resume_id not in seen:
            seen.add(resume_id)
            resume_ids.append(resume_id)

    if not resume_ids:
        raise ValueError("resume_ids cannot be empty")
    return resume_ids


@router.get("/")
def index():
    """Health check and API navigation."""
    return JSONResponse({
        "message": "ok",
        "docs": "/docs",
        "endpoints": [
            "/api/parse",
            "/api/parse/batch",
            "/api/job-context",
            "/api/query-rewrite",
            "/api/hard_filter_sql",
            "/api/vector_retrieve",
            "/api/rag-search",
            "/api/scoring-schema",
            "/api/score-resumes",
            "/api/scoring-feedback",
            "/api/scoring-feedback/batch",
            "/api/scoring-search",
        ],
    })


@router.post("/api/parse", dependencies=[Depends(require_internal)])
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


@router.post("/api/parse/batch", dependencies=[Depends(require_internal)])
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
@router.post("/api/job-context", dependencies=[Depends(require_hr_or_internal)])
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


@router.post("/api/query-rewrite", dependencies=[Depends(require_hr_or_internal)])
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


@router.post("/api/hard_filter_sql", dependencies=[Depends(require_hr_or_internal)])
async def hard_filter_sql(payload: dict):
    """Apply hard_filters to Postgres and return matching resume IDs."""
    try:
        hard_filters = HardFilters.model_validate(payload)
        resume_ids = query_resume_ids_by_hard_filters(hard_filters)
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse({
        "hard_filters": hard_filters.model_dump(),
        "resume_ids": resume_ids,
        "count": len(resume_ids),
    })


@router.post("/api/vector_retrieve", dependencies=[Depends(require_hr_or_internal)])
async def vector_retrieve(payload: dict):
    """Rank candidate resumes by vector similarity inside a filtered candidate pool."""
    try:
        request_model = VectorRetrieveRequest.model_validate(payload)
        results = query_similar_resumes(
            resume_ids=request_model.resume_ids,
            search_query_embedding=request_model.search_query_embedding,
            top_k=request_model.top_k,
        )
        response = VectorRetrieveResponse(
            search_query=request_model.search_query,
            top_k=request_model.top_k,
            count=len(results),
            results=results,
        )
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse(response.model_dump())


@router.post("/api/rag-search", dependencies=[Depends(require_hr_or_internal)])
async def rag_search(
    hr_note: str = Form(""),
    jd_text: str = Form(""),
    jd_file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
):
    """End-to-end retrieval pipeline: job context -> query rewrite -> hard filter -> vector retrieval."""
    if top_k <= 0:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer",
        )

    jd_file_content: Optional[bytes] = None
    if jd_file is not None:
        raw = await jd_file.read()
        if raw:
            jd_file_content = raw

    try:
        jd_body = resolve_jd_body(jd_text, jd_file_content)
        context_result = build_job_context(hr_note, jd_body)
        query, query_usage = rewrite_merged_context(context_result["merged_context"])
        search_query_embedding = embed_search_query(query)
        if not search_query_embedding:
            raise DatabaseError("search_query_embedding is empty")

        filtered_resume_ids = query_resume_ids_by_hard_filters(query.hard_filters)
        vector_results = query_similar_resumes(
            resume_ids=filtered_resume_ids,
            search_query_embedding=search_query_embedding,
            top_k=top_k,
        )
    except JDSourceConflict as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobContextEmpty as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse({
        "hard_filters": query.hard_filters.model_dump(),
        "search_query": query.search_query,
        "search_query_embedding": search_query_embedding,
        "filtered_resume_ids": filtered_resume_ids,
        "top_k": top_k,
        "top_k_resume_ids": [item.resume_id for item in vector_results],
        "count": len(vector_results),
        "query_usage": query_usage,
    })


@router.post("/api/scoring-schema", dependencies=[Depends(require_internal)])
async def create_scoring_schema(
    schema_name: str = Form(...),
    rules: str = Form(...),
):
    """Create a scoring schema from rules text, generate summary + embedding, and persist it."""
    schema_id = uuid4().hex
    try:
        payload, usage = build_scoring_schema_payload(schema_name, rules)
        persisted = save_scoring_schema(
            schema_id=schema_id,
            schema_name=payload["schema_name"],
            rules_json=payload["rules_json"],
            summary=payload["summary"],
            embedding=payload["embedding"],
        )
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse({
        **persisted,
        "summary_embedding_generated": payload["embedding"] is not None,
        "usage": usage,
    })


@router.post("/api/score-resumes")
async def score_resumes(
    jd_file: UploadFile = File(...),
    resume_ids: str = Form(...),
    hr_note: str = Form(""),
    feedback_examples_per_label: int = Form(2),
):
    """Score manually selected resumes using the best-matching scoring schema.

    The JD PDF is rewritten into a semantic query and embedded. That embedding is
    matched against active scoring schemas. The selected schema, optional feedback
    examples, and each target resume are then sent to the LLM for scoring.
    """
    if feedback_examples_per_label < 0:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="feedback_examples_per_label must be >= 0",
        )

    try:
        selected_resume_ids = _parse_resume_ids(resume_ids)
        jd_file_content = await jd_file.read()
        if not jd_file_content:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=ERR_FILE_EMPTY)

        jd_body = resolve_jd_body("", jd_file_content)
        context_result = build_job_context(hr_note, jd_body)
        query, query_usage = rewrite_merged_context(context_result["merged_context"])
        search_query_embedding = embed_search_query(query)
        if not search_query_embedding:
            raise DatabaseError("search_query_embedding is empty")

        schema = find_best_scoring_schema(search_query_embedding)
        feedback_examples = get_feedback_examples(
            schema_id=schema["schema_id"],
            limit_per_label=feedback_examples_per_label,
        )
        resumes = get_resumes_by_ids(selected_resume_ids)
        found_ids = {resume["resume_id"] for resume in resumes}
        missing_resume_ids = [
            resume_id for resume_id in selected_resume_ids if resume_id not in found_ids
        ]

        results = []
        scoring_usage = []
        for resume in resumes:
            score, usage = score_resume_with_schema(
                schema=schema,
                feedback_examples=feedback_examples,
                resume=resume,
            )
            results.append(score.model_dump())
            scoring_usage.append({
                "resume_id": resume["resume_id"],
                "usage": usage,
            })
    except HTTPException:
        raise
    except JDSourceConflict as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobContextEmpty as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse({
        "schema": schema,
        "feedback_examples_used": feedback_examples,
        "feedback_examples_count": len(feedback_examples),
        "feedback_examples_empty": len(feedback_examples) == 0,
        "search_query": query.search_query,
        "search_query_embedding": search_query_embedding,
        "requested_resume_ids": selected_resume_ids,
        "missing_resume_ids": missing_resume_ids,
        "count": len(results),
        "results": results,
        "query_usage": query_usage,
        "scoring_usage": scoring_usage,
    })


@router.post("/api/scoring-feedback")
async def create_scoring_feedback(payload: dict):
    """Persist human feedback for one scoring result."""
    try:
        feedback = save_scoring_feedback(
            feedback_id=uuid4().hex,
            schema_id=str(payload.get("schema_id", "")).strip(),
            resume_id=str(payload.get("resume_id", "")).strip(),
            label=str(payload.get("label", "")).strip().lower(),
            feedback_text=payload.get("feedback_text"),
            score=payload.get("score"),
            scoring_result=payload.get("scoring_result"),
        )
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse(feedback)


@router.post("/api/scoring-feedback/batch")
async def create_scoring_feedback_batch(payload: dict):
    """Persist human feedback for multiple scoring results."""
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="items must be a non-empty list",
        )

    saved, failed = [], []
    for item in items:
        if not isinstance(item, dict):
            failed.append({"item": item, "reason": "item must be an object"})
            continue
        try:
            feedback = save_scoring_feedback(
                feedback_id=uuid4().hex,
                schema_id=str(item.get("schema_id", "")).strip(),
                resume_id=str(item.get("resume_id", "")).strip(),
                label=str(item.get("label", "")).strip().lower(),
                feedback_text=item.get("feedback_text"),
                score=item.get("score"),
                scoring_result=item.get("scoring_result"),
            )
            saved.append(feedback)
        except Exception as exc:
            failed.append({
                "schema_id": item.get("schema_id"),
                "resume_id": item.get("resume_id"),
                "reason": str(exc),
            })

    return JSONResponse({
        "total": len(items),
        "saved_count": len(saved),
        "failed_count": len(failed),
        "saved": saved,
        "failed": failed,
    })


@router.post("/api/scoring-search")
async def scoring_search(
    hr_note: str = Form(""),
    jd_text: str = Form(""),
    jd_file: Optional[UploadFile] = File(None),
    initial_top_k: int = Form(10),
    feedback_examples_per_label: int = Form(2),
):
    """Full pipeline: JD/HR note -> retrieval -> schema selection -> LLM scoring."""
    if initial_top_k <= 0:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="initial_top_k must be a positive integer",
        )
    if feedback_examples_per_label < 0:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="feedback_examples_per_label must be >= 0",
        )

    jd_file_content: Optional[bytes] = None
    if jd_file is not None:
        raw = await jd_file.read()
        if raw:
            jd_file_content = raw

    try:
        jd_body = resolve_jd_body(jd_text, jd_file_content)
        context_result = build_job_context(hr_note, jd_body)
        query, query_usage = rewrite_merged_context(context_result["merged_context"])
        search_query_embedding = embed_search_query(query)
        if not search_query_embedding:
            raise DatabaseError("search_query_embedding is empty")

        filtered_resume_ids = query_resume_ids_by_hard_filters(query.hard_filters)
        retrieval_results = query_similar_resumes(
            resume_ids=filtered_resume_ids,
            search_query_embedding=search_query_embedding,
            top_k=initial_top_k,
        )
        retrieved_resume_ids = [item.resume_id for item in retrieval_results]
        retrieval_by_id = {
            item.resume_id: item.model_dump()
            for item in retrieval_results
        }

        schema = find_best_scoring_schema(search_query_embedding)
        feedback_examples = get_feedback_examples(
            schema_id=schema["schema_id"],
            limit_per_label=feedback_examples_per_label,
        )
        resumes = get_resumes_by_ids(retrieved_resume_ids)

        scored_results = []
        scoring_usage = []
        for resume in resumes:
            score, usage = score_resume_with_schema(
                schema=schema,
                feedback_examples=feedback_examples,
                resume=resume,
            )
            result = score.model_dump()
            result["retrieval"] = retrieval_by_id.get(resume["resume_id"], {})
            scored_results.append(result)
            scoring_usage.append({
                "resume_id": resume["resume_id"],
                "usage": usage,
            })

        scored_results.sort(key=lambda item: item["score"], reverse=True)
    except JDSourceConflict as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobContextEmpty as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse({
        "hard_filters": query.hard_filters.model_dump(),
        "search_query": query.search_query,
        "search_query_embedding": search_query_embedding,
        "filtered_resume_ids": filtered_resume_ids,
        "initial_top_k": initial_top_k,
        "retrieved_resume_ids": retrieved_resume_ids,
        "schema": schema,
        "feedback_examples_used": feedback_examples,
        "feedback_examples_count": len(feedback_examples),
        "feedback_examples_empty": len(feedback_examples) == 0,
        "count": len(scored_results),
        "results": scored_results,
        "query_usage": query_usage,
        "scoring_usage": scoring_usage,
    })
