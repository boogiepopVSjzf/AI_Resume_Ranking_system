from __future__ import annotations

from typing import Any

from schemas.models import ResumeStructured
from services.embedding_service import embed_text
from services.resume_rag_split import _build_metadata, _build_full_text


def build_resume_storage_bundle(resume: ResumeStructured) -> dict[str, Any]:
    """Produce the unified output record for a parsed resume.

    Returns ``metadata``, ``semantic_text``, ``embedding``, and ``raw_json``
    as the single canonical payload shape.
    """
    metadata = _build_metadata(resume)
    semantic_text = _build_full_text(resume)
    embedding = embed_text(semantic_text)
    raw_json = resume.model_dump()

    return {
        "metadata": metadata,
        "semantic_text": semantic_text,
        "embedding": embedding,
        "raw_json": raw_json,
    }
