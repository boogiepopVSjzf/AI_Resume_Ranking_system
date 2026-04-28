import hashlib
import json
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
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
from services.job_query_rewrite_service import (
    embed_search_query,
    filter_mode_to_prompt_variant,
    rewrite_merged_context,
)
from services.resume_storage_bundle import build_resume_storage_bundle
from services.scoring_schema_service import build_scoring_schema_payload, parse_rules_text
from services.scoring_service import (
    ResumeScore,
    build_feedback_calibration_data,
    normalize_feedback_influence_mode,
    score_resume_with_schema,
)
from services.reranker_service import rerank_resumes
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
    RerankerError,
)
from utils.logger import get_logger
from schemas.job_query import (
    HardFilters,
    StandardizedJobQuery,
    VectorRetrieveRequest,
    VectorRetrieveResponse,
)
from schemas.models import ExtractionInput

router = APIRouter()
logger = get_logger("api")
_S3_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_QUERY_REWRITE_CACHE: OrderedDict[str, dict] = OrderedDict()
_SCORING_CACHE: OrderedDict[str, dict] = OrderedDict()
_SCHEMA_CACHE: OrderedDict[str, dict] = OrderedDict()


def _normalize_filter_mode(raw_mode: str) -> str:
    mode = (raw_mode or "balanced").strip().lower()
    if mode not in {"strict", "balanced", "semantic_only"}:
        raise ValueError("filter_mode must be one of: strict, balanced, semantic_only")
    return mode


def _apply_filter_mode(hard_filters: HardFilters, filter_mode: str) -> HardFilters:
    """Decide which hard filters reach Postgres.

    Strictness is controlled at the prompt level: "strict" uses the permissive
    rewrite prompt while "balanced" uses the conservative one (see
    services/job_query_rewrite_service.py). Here we only need to disable hard
    filtering entirely when the caller explicitly opts into "semantic_only".
    """
    mode = _normalize_filter_mode(filter_mode)
    if mode == "semantic_only":
        return HardFilters()
    return hard_filters.model_copy(deep=True)


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _retrieval_pool_size(final_top_k: int) -> int:
    if final_top_k <= 0:
        raise ValueError("final_top_k must be a positive integer")
    if not settings.ENABLE_RERANKER:
        return final_top_k
    return max(final_top_k, settings.RERANKER_CANDIDATE_POOL_SIZE)

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
    RerankerError: HTTP_502_BAD_GATEWAY,
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


def _normalize_cache_context(merged_context: str) -> str:
    return "\n".join(line.rstrip() for line in merged_context.strip().splitlines())


def _parse_query_model_for_cache(provider: str, model: Optional[str]) -> str:
    if model:
        return model
    provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower().strip()
    fallback_model = {
        "anthropic": settings.ANTHROPIC_MODEL,
        "dashscope": settings.LLM_MODEL,
        "gemini": settings.GEMINI_MODEL,
        "openai": settings.OPENAI_MODEL,
        "ollama": settings.OLLAMA_MODEL,
    }.get(provider, settings.DEFAULT_LLM_MODEL)
    return fallback_model


def _resolved_model_for_cache(provider: str, model: Optional[str]) -> str:
    if model:
        return model
    provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower().strip()
    fallback_model = {
        "anthropic": settings.ANTHROPIC_MODEL,
        "dashscope": settings.LLM_MODEL,
        "gemini": settings.GEMINI_MODEL,
        "openai": settings.OPENAI_MODEL,
        "ollama": settings.OLLAMA_MODEL,
    }.get(provider, settings.DEFAULT_LLM_MODEL)
    return fallback_model


def _query_rewrite_cache_key(merged_context: str, filter_mode: str) -> str:
    provider = settings.PARSE_QUERY_LLM_PROVIDER
    payload = {
        "merged_context": _normalize_cache_context(merged_context),
        "provider": provider,
        "model": _parse_query_model_for_cache(provider, settings.PARSE_QUERY_LLM_MODEL),
        "prompt_variant": filter_mode_to_prompt_variant(filter_mode),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _rewrite_merged_context_cached(
    merged_context: str,
    filter_mode: str = "balanced",
) -> tuple[StandardizedJobQuery, dict]:
    """Reuse deterministic query-rewrite output for identical JD/HR inputs.

    The cache key incorporates the prompt variant chosen by ``filter_mode``,
    so "strict" and "balanced" do not share entries while "balanced" and
    "semantic_only" do (they use the same conservative prompt).
    """
    mode = _normalize_filter_mode(filter_mode)
    if not settings.QUERY_REWRITE_CACHE_ENABLED:
        return rewrite_merged_context(merged_context, filter_mode=mode)

    cache_key = _query_rewrite_cache_key(merged_context, mode)
    cached = _QUERY_REWRITE_CACHE.get(cache_key)
    if cached is not None:
        _QUERY_REWRITE_CACHE.move_to_end(cache_key)
        usage = dict(cached.get("usage", {}))
        usage.update({"cached": True, "cache_key": cache_key})
        return StandardizedJobQuery.model_validate(cached["query"]), usage

    query, usage = rewrite_merged_context(merged_context, filter_mode=mode)
    _QUERY_REWRITE_CACHE[cache_key] = {
        "query": query.model_dump(),
        "usage": dict(usage or {}),
    }
    _QUERY_REWRITE_CACHE.move_to_end(cache_key)

    max_size = max(settings.QUERY_REWRITE_CACHE_MAX_SIZE, 1)
    while len(_QUERY_REWRITE_CACHE) > max_size:
        _QUERY_REWRITE_CACHE.popitem(last=False)

    response_usage = dict(usage or {})
    response_usage.update({"cached": False, "cache_key": cache_key})
    return query, response_usage


def _scoring_cache_key(
    *,
    schema: dict,
    feedback_examples: list[dict],
    resume: dict,
    feedback_influence_mode: str,
) -> str:
    provider = settings.SCORING_LLM_PROVIDER
    payload = {
        "prompt_version": settings.SCORING_PROMPT_VERSION,
        "provider": provider,
        "model": _resolved_model_for_cache(provider, settings.SCORING_LLM_MODEL),
        "schema": {
            "schema_id": schema.get("schema_id"),
            "schema_name": schema.get("schema_name"),
            "version": schema.get("version"),
            "summary": schema.get("summary"),
            "rules_json": schema.get("rules_json"),
        },
        "feedback_examples": feedback_examples,
        "feedback_influence_mode": feedback_influence_mode,
        "resume": {
            "resume_id": resume.get("resume_id"),
            "metadata": resume.get("metadata", {}),
            "semantic_text": resume.get("semantic_text", ""),
            "raw_json": resume.get("raw_json", {}),
        },
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _schema_cache_key(*, schema_name: str, rules_json: dict) -> str:
    provider = settings.SCHEMA_LLM_PROVIDER
    payload = {
        "prompt_version": settings.SCHEMA_PROMPT_VERSION,
        "schema_name": schema_name.strip(),
        "rules_json": rules_json,
        "provider": provider,
        "model": _resolved_model_for_cache(provider, settings.SCHEMA_LLM_MODEL),
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _create_scoring_schema_cached(*, schema_name: str, rules: str) -> dict:
    """Reuse a created schema response for identical Schema Studio inputs."""
    normalized_name = schema_name.strip()
    if not normalized_name:
        raise ValueError("schema_name cannot be empty")

    rules_json = parse_rules_text(rules)
    cache_key = _schema_cache_key(schema_name=normalized_name, rules_json=rules_json)

    if settings.SCHEMA_CACHE_ENABLED:
        cached = _SCHEMA_CACHE.get(cache_key)
        if cached is not None:
            _SCHEMA_CACHE.move_to_end(cache_key)
            response = dict(cached["response"])
            usage = dict(response.get("usage", {}))
            usage.update({"cached": True, "cache_key": cache_key})
            response["usage"] = usage
            response["schema_cache_reused"] = True
            return response

    schema_id = uuid4().hex
    payload, usage = build_scoring_schema_payload(normalized_name, rules)
    persisted = save_scoring_schema(
        schema_id=schema_id,
        schema_name=payload["schema_name"],
        rules_json=payload["rules_json"],
        summary=payload["summary"],
        embedding=payload["embedding"],
    )
    response_usage = dict(usage or {})
    response_usage.update({"cached": False, "cache_key": cache_key})
    response = {
        **persisted,
        "summary_embedding_generated": payload["embedding"] is not None,
        "usage": response_usage,
        "schema_cache_reused": False,
    }

    if settings.SCHEMA_CACHE_ENABLED:
        _SCHEMA_CACHE[cache_key] = {"response": response}
        _SCHEMA_CACHE.move_to_end(cache_key)

        max_size = max(settings.SCHEMA_CACHE_MAX_SIZE, 1)
        while len(_SCHEMA_CACHE) > max_size:
            _SCHEMA_CACHE.popitem(last=False)

    return response


def _score_resume_with_schema_cached(
    *,
    schema: dict,
    feedback_examples: list[dict],
    resume: dict,
    feedback_influence_mode: str = "on",
) -> tuple[ResumeScore, dict]:
    """Reuse LLM scoring output for identical resume/schema/feedback inputs."""
    feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    if not settings.SCORING_CACHE_ENABLED:
        return score_resume_with_schema(
            schema=schema,
            feedback_examples=feedback_examples,
            resume=resume,
            feedback_influence_mode=feedback_influence_mode,
        )

    cache_key = _scoring_cache_key(
        schema=schema,
        feedback_examples=feedback_examples,
        resume=resume,
        feedback_influence_mode=feedback_influence_mode,
    )
    cached = _SCORING_CACHE.get(cache_key)
    if cached is not None:
        _SCORING_CACHE.move_to_end(cache_key)
        usage = dict(cached.get("usage", {}))
        usage.update({"cached": True, "cache_key": cache_key})
        return ResumeScore.model_validate(cached["score"]), usage

    score, usage = score_resume_with_schema(
        schema=schema,
        feedback_examples=feedback_examples,
        resume=resume,
        feedback_influence_mode=feedback_influence_mode,
    )
    _SCORING_CACHE[cache_key] = {
        "score": score.model_dump(),
        "usage": dict(usage or {}),
    }
    _SCORING_CACHE.move_to_end(cache_key)

    max_size = max(settings.SCORING_CACHE_MAX_SIZE, 1)
    while len(_SCORING_CACHE) > max_size:
        _SCORING_CACHE.popitem(last=False)

    response_usage = dict(usage or {})
    response_usage.update({"cached": False, "cache_key": cache_key})
    return score, response_usage


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


def _configured(value: Optional[str]) -> bool:
    return bool(value and str(value).strip())


def _provider_key_configured(provider: str) -> bool:
    provider = (provider or "").lower().strip()
    if provider == "dashscope":
        return _configured(settings.LLM_API_KEY)
    if provider == "gemini":
        return _configured(settings.GEMINI_API_KEY)
    if provider == "openai":
        return _configured(settings.OPENAI_API_KEY)
    if provider == "anthropic":
        return _configured(settings.ANTHROPIC_API_KEY)
    if provider == "ollama":
        return True
    return False


def _role_llm_status(label: str, provider: str, model: Optional[str]) -> dict:
    provider = (provider or settings.DEFAULT_LLM_PROVIDER).lower().strip()
    fallback_model = {
        "anthropic": settings.ANTHROPIC_MODEL,
        "dashscope": settings.LLM_MODEL,
        "gemini": settings.GEMINI_MODEL,
        "openai": settings.OPENAI_MODEL,
        "ollama": settings.OLLAMA_MODEL,
    }.get(provider, settings.DEFAULT_LLM_MODEL)
    return {
        "label": label,
        "provider": provider,
        "model": model or fallback_model,
        "api_key_configured": _provider_key_configured(provider),
        "temperature": settings.LLM_TEMPERATURE,
    }


def _build_system_warnings(status: dict) -> list[str]:
    warnings = []
    if status["database"]["enabled"] and not status["database"]["url_configured"]:
        warnings.append("Database persistence is enabled but DATABASE_URL is missing.")
    if status["s3"]["enabled"]:
        if not status["s3"]["bucket_configured"]:
            warnings.append("S3 storage is enabled but S3_BUCKET_NAME is missing.")
        if not status["s3"]["credentials_configured"]:
            warnings.append("S3 storage is enabled but AWS credentials are incomplete.")
        if not status["s3"]["region_configured"]:
            warnings.append("S3 storage is enabled but AWS_REGION is missing.")
    for role, llm_status in status["llm_routing"].items():
        if not llm_status["api_key_configured"]:
            warnings.append(f"{role} uses {llm_status['provider']} but its API key is missing.")
    return warnings


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
            "/api/system/status",
        ],
    })


@router.get("/api/health")
def health_check():
    """Stable health endpoint for the React frontend."""
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
            "/api/system/status",
        ],
    })


@router.get("/api/system/status")
def system_status():
    """Return safe, read-only runtime configuration status for the frontend."""
    status = {
        "runtime": {
            "mode": "read_only",
            "env_loaded_from": str(settings.BASE_DIR / ".env"),
            "restart_required_after_env_change": True,
        },
        "database": {
            "enabled": settings.ENABLE_DB_PERSISTENCE,
            "url_configured": _configured(settings.DATABASE_URL),
            "sslmode": settings.DATABASE_SSLMODE,
            "auto_init": settings.DB_AUTO_INIT,
        },
        "s3": {
            "enabled": settings.ENABLE_S3_STORAGE,
            "bucket_configured": _configured(settings.S3_BUCKET_NAME),
            "bucket": settings.S3_BUCKET_NAME if settings.S3_BUCKET_NAME else None,
            "region_configured": _configured(settings.AWS_REGION),
            "region": settings.AWS_REGION if settings.AWS_REGION else None,
            "endpoint_configured": _configured(settings.S3_ENDPOINT_URL),
            "credentials_configured": _configured(settings.AWS_ACCESS_KEY_ID)
            and _configured(settings.AWS_SECRET_ACCESS_KEY),
        },
        "llm_routing": {
            "parse_query": _role_llm_status(
                "Parse + Query",
                settings.PARSE_QUERY_LLM_PROVIDER,
                settings.PARSE_QUERY_LLM_MODEL,
            ),
            "schema": _role_llm_status(
                "Schema",
                settings.SCHEMA_LLM_PROVIDER,
                settings.SCHEMA_LLM_MODEL,
            ),
            "scoring": _role_llm_status(
                "Scoring",
                settings.SCORING_LLM_PROVIDER,
                settings.SCORING_LLM_MODEL,
            ),
        },
        "llm_stability": {
            "temperature": settings.LLM_TEMPERATURE,
            "schema_cache_enabled": settings.SCHEMA_CACHE_ENABLED,
            "schema_cache_max_size": settings.SCHEMA_CACHE_MAX_SIZE,
            "schema_cache_size": len(_SCHEMA_CACHE),
            "schema_prompt_version": settings.SCHEMA_PROMPT_VERSION,
            "query_rewrite_cache_enabled": settings.QUERY_REWRITE_CACHE_ENABLED,
            "query_rewrite_cache_max_size": settings.QUERY_REWRITE_CACHE_MAX_SIZE,
            "query_rewrite_cache_size": len(_QUERY_REWRITE_CACHE),
            "scoring_cache_enabled": settings.SCORING_CACHE_ENABLED,
            "scoring_cache_max_size": settings.SCORING_CACHE_MAX_SIZE,
            "scoring_cache_size": len(_SCORING_CACHE),
            "scoring_prompt_version": settings.SCORING_PROMPT_VERSION,
        },
        "embedding": {
            "model": settings.EMBEDDING_MODEL,
            "device": settings.EMBEDDING_DEVICE,
            "dimension": settings.EMBEDDING_DIMENSION,
            "preload": settings.PRELOAD_EMBEDDING_MODEL,
            "include_embedding_in_response": settings.INCLUDE_EMBEDDING_IN_RESPONSE,
        },
        "reranker": {
            "enabled": settings.ENABLE_RERANKER,
            "model": settings.RERANKER_MODEL,
            "device": settings.RERANKER_DEVICE,
            "preload": settings.PRELOAD_RERANKER_MODEL,
            "candidate_pool_size": settings.RERANKER_CANDIDATE_POOL_SIZE,
        },
        "limits": {
            "allowed_extensions": sorted(settings.ALLOWED_EXTENSIONS),
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "max_batch_size": MAX_BATCH_SIZE,
        },
    }
    status["warnings"] = _build_system_warnings(status)
    return JSONResponse(status)


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
            query, rewrite_usage = _rewrite_merged_context_cached(
                result["merged_context"], "balanced"
            )
            result["standardized_query"] = query.model_dump()
            result["search_query_embedding"] = embed_search_query(query)
            result["query_rewrite_usage"] = rewrite_usage
        except Exception as exc:
            _raise_http_exception(exc)

    return JSONResponse(result)


@router.post("/api/query-rewrite")
async def query_rewrite(payload: dict):
    """Rewrite merged_context into hard_filters + search_query via LLM.

    The optional ``filter_mode`` field selects which rewrite prompt is used:
    "strict" (permissive) or "balanced"/"semantic_only" (conservative). Defaults
    to "balanced".
    """
    merged_context = payload.get("merged_context")
    if not isinstance(merged_context, str) or not merged_context.strip():
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="merged_context is required and must be a non-empty string",
        )

    try:
        filter_mode = _normalize_filter_mode(payload.get("filter_mode") or "balanced")
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        query, usage = _rewrite_merged_context_cached(merged_context, filter_mode)
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


@router.post("/api/hard_filter_sql")
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


@router.post("/api/vector_retrieve")
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


@router.post("/api/rag-search")
async def rag_search(
    hr_note: str = Form(""),
    jd_text: str = Form(""),
    jd_file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
    filter_mode: str = Form("balanced"),
):
    """End-to-end retrieval pipeline: job context -> query rewrite -> hard filter -> vector retrieval."""
    if top_k <= 0:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer",
        )
    try:
        filter_mode = _normalize_filter_mode(filter_mode)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    jd_file_content: Optional[bytes] = None
    if jd_file is not None:
        raw = await jd_file.read()
        if raw:
            jd_file_content = raw

    try:
        jd_body = resolve_jd_body(jd_text, jd_file_content)
        context_result = build_job_context(hr_note, jd_body)
        query, query_usage = _rewrite_merged_context_cached(
            context_result["merged_context"], filter_mode
        )
        search_query_embedding = embed_search_query(query)
        if not search_query_embedding:
            raise DatabaseError("search_query_embedding is empty")

        applied_hard_filters = _apply_filter_mode(query.hard_filters, filter_mode)
        filtered_resume_ids = query_resume_ids_by_hard_filters(applied_hard_filters)
        retrieval_pool_size = _retrieval_pool_size(top_k)
        vector_results = query_similar_resumes(
            resume_ids=filtered_resume_ids,
            search_query_embedding=search_query_embedding,
            top_k=retrieval_pool_size,
        )
        vector_pool_resume_ids = [item.resume_id for item in vector_results]
        reranked_results = vector_results
        if settings.ENABLE_RERANKER and vector_results:
            rerank_resumes_payload = get_resumes_by_ids(vector_pool_resume_ids)
            reranked_payload = rerank_resumes(
                search_query=query.search_query,
                resumes=rerank_resumes_payload,
                top_k=top_k,
            )
            reranker_scores_by_id = {
                item["resume_id"]: item.get("reranker_score")
                for item in reranked_payload
            }
            vector_by_id = {item.resume_id: item for item in vector_results}
            reranked_results = []
            for item in reranked_payload:
                vector_item = vector_by_id.get(item["resume_id"])
                if vector_item is None:
                    continue
                reranked_results.append(
                    vector_item.model_copy(
                        update={"reranker_score": reranker_scores_by_id.get(item["resume_id"])}
                    )
                )
        else:
            reranked_results = vector_results[:top_k]
    except JDSourceConflict as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobContextEmpty as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse(_json_safe({
        "hard_filters": query.hard_filters.model_dump(),
        "applied_hard_filters": applied_hard_filters.model_dump(),
        "filter_mode": filter_mode,
        "search_query": query.search_query,
        "search_query_embedding": search_query_embedding,
        "filtered_resume_ids": filtered_resume_ids,
        "retrieval_pool_size": retrieval_pool_size,
        "reranker_enabled": settings.ENABLE_RERANKER,
        "reranker_model": settings.RERANKER_MODEL if settings.ENABLE_RERANKER else None,
        "vector_candidate_pool_ids": vector_pool_resume_ids,
        "top_k": top_k,
        "top_k_resume_ids": [item.resume_id for item in reranked_results],
        "count": len(reranked_results),
        "query_usage": query_usage,
    }))


@router.post("/api/scoring-schema")
async def create_scoring_schema(
    schema_name: str = Form(...),
    rules: str = Form(...),
):
    """Create a scoring schema from rules text, generate summary + embedding, and persist it."""
    try:
        response = _create_scoring_schema_cached(schema_name=schema_name, rules=rules)
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse(response)


@router.post("/api/score-resumes")
async def score_resumes(
    jd_file: UploadFile = File(...),
    resume_ids: str = Form(...),
    hr_note: str = Form(""),
    feedback_examples_per_label: int = Form(2),
    feedback_influence_mode: str = Form("on"),
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
        feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        selected_resume_ids = _parse_resume_ids(resume_ids)
        jd_file_content = await jd_file.read()
        if not jd_file_content:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=ERR_FILE_EMPTY)

        jd_body = resolve_jd_body("", jd_file_content)
        context_result = build_job_context(hr_note, jd_body)
        query, query_usage = _rewrite_merged_context_cached(
            context_result["merged_context"], "balanced"
        )
        search_query_embedding = embed_search_query(query)
        if not search_query_embedding:
            raise DatabaseError("search_query_embedding is empty")

        schema = find_best_scoring_schema(search_query_embedding)
        feedback_examples = get_feedback_examples(
            schema_id=schema["schema_id"],
            limit_per_label=feedback_examples_per_label,
        )
        feedback_calibration_data = build_feedback_calibration_data(feedback_examples)
        resumes = get_resumes_by_ids(selected_resume_ids)
        found_ids = {resume["resume_id"] for resume in resumes}
        missing_resume_ids = [
            resume_id for resume_id in selected_resume_ids if resume_id not in found_ids
        ]

        results = []
        scoring_usage = []
        for resume in resumes:
            score, usage = _score_resume_with_schema_cached(
                schema=schema,
                feedback_examples=feedback_examples,
                resume=resume,
                feedback_influence_mode=feedback_influence_mode,
            )
            result = score.model_dump()
            result["candidate_name"] = resume.get("candidate_name") or ""
            results.append(result)
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
        "feedback_calibration_data": feedback_calibration_data,
        "feedback_influence_mode": feedback_influence_mode,
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
    filter_mode: str = Form("balanced"),
    feedback_examples_per_label: int = Form(2),
    feedback_influence_mode: str = Form("on"),
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
    try:
        filter_mode = _normalize_filter_mode(filter_mode)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    try:
        feedback_influence_mode = normalize_feedback_influence_mode(feedback_influence_mode)
    except ValueError as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    jd_file_content: Optional[bytes] = None
    if jd_file is not None:
        raw = await jd_file.read()
        if raw:
            jd_file_content = raw

    try:
        jd_body = resolve_jd_body(jd_text, jd_file_content)
        context_result = build_job_context(hr_note, jd_body)
        query, query_usage = _rewrite_merged_context_cached(
            context_result["merged_context"], filter_mode
        )
        search_query_embedding = embed_search_query(query)
        if not search_query_embedding:
            raise DatabaseError("search_query_embedding is empty")

        applied_hard_filters = _apply_filter_mode(query.hard_filters, filter_mode)
        filtered_resume_ids = query_resume_ids_by_hard_filters(applied_hard_filters)
        retrieval_pool_size = _retrieval_pool_size(initial_top_k)
        retrieval_results = query_similar_resumes(
            resume_ids=filtered_resume_ids,
            search_query_embedding=search_query_embedding,
            top_k=retrieval_pool_size,
        )
        vector_candidate_pool_ids = [item.resume_id for item in retrieval_results]
        retrieval_by_id = {
            item.resume_id: item.model_dump()
            for item in retrieval_results
        }
        retrieved_resume_ids = vector_candidate_pool_ids
        if settings.ENABLE_RERANKER and retrieval_results:
            rerank_resumes_payload = get_resumes_by_ids(vector_candidate_pool_ids)
            reranked_payload = rerank_resumes(
                search_query=query.search_query,
                resumes=rerank_resumes_payload,
                top_k=initial_top_k,
            )
            retrieved_resume_ids = [item["resume_id"] for item in reranked_payload]
            reranker_scores_by_id = {
                item["resume_id"]: item.get("reranker_score")
                for item in reranked_payload
            }
            for resume_id, score in reranker_scores_by_id.items():
                if resume_id in retrieval_by_id:
                    retrieval_by_id[resume_id]["reranker_score"] = score

        schema = find_best_scoring_schema(search_query_embedding)
        feedback_examples = get_feedback_examples(
            schema_id=schema["schema_id"],
            limit_per_label=feedback_examples_per_label,
        )
        feedback_calibration_data = build_feedback_calibration_data(feedback_examples)
        resumes = get_resumes_by_ids(retrieved_resume_ids)

        scored_results = []
        scoring_usage = []
        for resume in resumes:
            score, usage = _score_resume_with_schema_cached(
                schema=schema,
                feedback_examples=feedback_examples,
                resume=resume,
                feedback_influence_mode=feedback_influence_mode,
            )
            result = score.model_dump()
            result["candidate_name"] = resume.get("candidate_name") or ""
            result["retrieval"] = retrieval_by_id.get(resume["resume_id"], {})
            scored_results.append(result)
            scoring_usage.append({
                "resume_id": resume["resume_id"],
                "usage": usage,
            })

        scored_results.sort(key=lambda item: (-item["score"], item["resume_id"]))
    except JDSourceConflict as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobContextEmpty as exc:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_http_exception(exc)

    return JSONResponse(_json_safe({
        "hard_filters": query.hard_filters.model_dump(),
        "applied_hard_filters": applied_hard_filters.model_dump(),
        "filter_mode": filter_mode,
        "search_query": query.search_query,
        "search_query_embedding": search_query_embedding,
        "filtered_resume_ids": filtered_resume_ids,
        "initial_top_k": initial_top_k,
        "retrieval_pool_size": retrieval_pool_size,
        "reranker_enabled": settings.ENABLE_RERANKER,
        "reranker_model": settings.RERANKER_MODEL if settings.ENABLE_RERANKER else None,
        "vector_candidate_pool_ids": vector_candidate_pool_ids,
        "retrieved_resume_ids": retrieved_resume_ids,
        "schema": schema,
        "feedback_examples_used": feedback_examples,
        "feedback_calibration_data": feedback_calibration_data,
        "feedback_influence_mode": feedback_influence_mode,
        "feedback_examples_count": len(feedback_examples),
        "feedback_examples_empty": len(feedback_examples) == 0,
        "count": len(scored_results),
        "results": scored_results,
        "query_usage": query_usage,
        "scoring_usage": scoring_usage,
    }))
