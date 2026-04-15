from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RuleSchemaRequest(BaseModel):
    rule: Dict[str, Any] = Field(..., description="JSON-format rule to describe and embed")
    job_name: str = Field(..., min_length=1, description="Job name for this schema")


class RuleDescriptionLLMOutput(BaseModel):
    rule_description: str


class RuleSchemaResponse(BaseModel):
    schema_id: str
    rule_json: Dict[str, Any]
    rule_description: str
    embedding_vector: List[float]
    job_name: str
