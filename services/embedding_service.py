from __future__ import annotations

import re
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger("embedding")

_model = None


def model_uses_e5_query_passage_prefixes(model_name: str) -> bool:
    """E5 (intfloat) retrieval checkpoints expect query: / passage: prefixes at encode time."""
    m = model_name.lower()
    if "multilingual-e5" in m:
        return True
    if "intfloat" in m and "e5" in m:
        return True
    if re.search(r"e5-(small|base|large)-v2", m):
        return True
    return False


def _prepare_input(text: str, *, for_query: bool) -> str:
    raw = text.strip()
    if not raw:
        return ""
    if model_uses_e5_query_passage_prefixes(settings.EMBEDDING_MODEL):
        prefix = "query: " if for_query else "passage: "
        if raw.lower().startswith(prefix.strip()):
            return raw
        return f"{prefix}{raw}"
    return raw


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading embedding model %s on %s",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE,
        )
        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
    return _model


def preload_embedding_model() -> None:
    """Warm the embedding model during app startup to avoid first-request latency."""
    _get_model()


def embed_text(text: str, *, for_query: bool = False) -> Optional[list[float]]:
    """Return a normalised embedding vector, or None for empty input.

    For intfloat E5 models, use for_query=True when embedding the job search string
    and for_query=False (default) when embedding resume text or schema summaries.
    """
    prepared = _prepare_input(text, for_query=for_query)
    if not prepared:
        return None
    vector = _get_model().encode(prepared, normalize_embeddings=True)
    return vector.tolist()
