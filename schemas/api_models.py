from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


ProviderName = Literal["gemini", "openai", "ollama"]


class LLMGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    provider: Optional[ProviderName] = None
    model: Optional[str] = None
