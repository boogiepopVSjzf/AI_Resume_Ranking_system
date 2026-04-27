from __future__ import annotations

import json
import math
from typing import Any

from config import settings
from utils.errors import RerankerError
from utils.logger import get_logger

logger = get_logger("reranker")

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is required for the local reranker."
            ) from exc

        logger.info(
            "Loading reranker model %s on %s",
            settings.RERANKER_MODEL,
            settings.RERANKER_DEVICE,
        )
        try:
            _model = CrossEncoder(
                settings.RERANKER_MODEL,
                device=settings.RERANKER_DEVICE,
            )
        except Exception as exc:
            raise RerankerError(f"Failed to load reranker model: {exc}") from exc
    return _model


def preload_reranker_model() -> None:
    """Warm the reranker model during app startup."""
    if not settings.ENABLE_RERANKER:
        return
    _get_model()


def _resume_text_for_reranker(resume: dict[str, Any]) -> str:
    semantic_text = str(resume.get("semantic_text") or "").strip()
    if semantic_text:
        return semantic_text

    metadata = resume.get("metadata")
    raw_json = resume.get("raw_json")
    chunks: list[str] = []
    if isinstance(metadata, dict) and metadata:
        chunks.append("Metadata:\n" + json.dumps(metadata, ensure_ascii=False))
    if isinstance(raw_json, dict) and raw_json:
        chunks.append("Raw resume data:\n" + json.dumps(raw_json, ensure_ascii=False))
    return "\n\n".join(chunks).strip()


def rerank_resumes(
    *,
    search_query: str,
    resumes: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rerank a retrieved candidate pool using a local cross-encoder reranker."""
    if top_k <= 0:
        raise RerankerError("top_k must be a positive integer")
    if not settings.ENABLE_RERANKER:
        return [
            {**resume, "reranker_score": None}
            for resume in resumes[:top_k]
        ]

    query = str(search_query or "").strip()
    if not query:
        raise RerankerError("search_query cannot be empty for reranking")

    scored_inputs: list[tuple[dict[str, Any], str]] = []
    for resume in resumes:
        candidate_text = _resume_text_for_reranker(resume)
        if candidate_text:
            scored_inputs.append((resume, candidate_text))

    if not scored_inputs:
        return []

    model = _get_model()
    pairs = [(query, text) for _, text in scored_inputs]
    try:
        scores = model.predict(pairs)
    except Exception as exc:
        raise RerankerError(f"Failed to rerank resumes: {exc}") from exc

    reranked: list[dict[str, Any]] = []
    for (resume, _), score in zip(scored_inputs, scores):
        numeric_score = float(score)
        safe_score = numeric_score if math.isfinite(numeric_score) else None
        reranked.append({
            **resume,
            "reranker_score": safe_score,
        })

    reranked.sort(
        key=lambda item: (
            -(item.get("reranker_score") or 0.0),
            item.get("resume_id") or "",
        )
    )
    return reranked[:top_k]
