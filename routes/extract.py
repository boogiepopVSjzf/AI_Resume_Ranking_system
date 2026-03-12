import json

import routes
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from schemas.api_models import ExtractResumeRequest
from schemas.models import ExtractionInput
from services.extract_service import extract_structured_resume
from storage.file_store import save_result_json
from storage.db_store import save_parsed_resume
from utils.errors import AppError, LLMParseError

routes = APIRouter(prefix="/api", tags=["extract"])


@routes.post("/extract")
async def extract_resume(payload: ExtractResumeRequest):
    extraction_input = ExtractionInput(text=payload.text)

    try:
        structured = extract_structured_resume(
            extraction_input,
            provider=payload.provider,
            model=payload.model,
        )
    except (LLMParseError, AppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    json_text = structured.model_dump_json(ensure_ascii=False)

    if isinstance(payload.resume_id, str) and payload.resume_id:
    save_result_json(payload.resume_id, json_text)
    try:
        save_parsed_resume(payload.resume_id, structured)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist parsed resume to database: {exc}",
        ) from exc

    return JSONResponse(json.loads(json_text))
