from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMScoringApiRequest(BaseModel):
    """
    Public API request for the LLM scoring endpoint.

    The caller only provides:
    - one or more resume JSON objects
    - one JD JSON object
    - a schema key that selects a preset scoring configuration
    """

    resumes: List[Dict[str, Any]] = Field(
        ...,
        description="One or more resume JSON objects from upstream retrieval.",
    )
    jd: Dict[str, Any] = Field(
        ...,
        description="One JD JSON object from upstream processing.",
    )
    schema_key: str = Field(
        ...,
        description="Preset scoring schema key, such as 'data_scientist_v1' or 'software_engineer_v1'.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional LLM provider override.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional LLM model override.",
    )