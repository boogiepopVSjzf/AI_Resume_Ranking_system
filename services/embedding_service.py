from __future__ import annotations

from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger("embedding")

_model = None


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
            settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE
        )
    return _model


def embed_text(text: str) -> Optional[list[float]]:
    """Return a normalised embedding vector, or None for empty input."""
    if not text or not text.strip():
        return None
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()
