from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from schemas.models import EducationLevel, MajorCategory


class HardFilters(BaseModel):
    min_yoe: Optional[int] = None
    required_skills: List[str] = Field(default_factory=list)
    education_level: Optional[EducationLevel] = None
    major: Optional[MajorCategory] = None


class StandardizedJobQuery(BaseModel):
    hard_filters: HardFilters
    search_query: str


class VectorRetrieveRequest(BaseModel):
    resume_ids: List[str] = Field(default_factory=list)
    search_query: Optional[str] = None
    search_query_embedding: List[float]
    top_k: int = 10


class VectorRetrieveResult(BaseModel):
    resume_id: str
    similarity_score: float
    metadata: dict = Field(default_factory=dict)
    raw_json: dict = Field(default_factory=dict)


class VectorRetrieveResponse(BaseModel):
    search_query: Optional[str] = None
    top_k: int
    count: int
    results: List[VectorRetrieveResult] = Field(default_factory=list)
