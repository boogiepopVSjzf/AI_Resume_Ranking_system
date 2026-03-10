import routes
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config import settings
from schemas.api_models import LLMGenerateRequest
from services.llm_service import call_llm
from utils.errors import LLMError

routes = APIRouter(prefix="/api/llm", tags=["llm"])


@routes.post("/generate")
async def generate_with_unified_llm(payload: LLMGenerateRequest):
    """
    Unified API endpoint for calling different models.
    """
    try:
        output = call_llm(
            prompt=payload.prompt,
            provider=payload.provider,
            model=payload.model,
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse(
        {
            "provider": payload.provider or settings.DEFAULT_LLM_PROVIDER,
            "model": payload.model or settings.DEFAULT_LLM_MODEL,
            "output": output,
        }
    )