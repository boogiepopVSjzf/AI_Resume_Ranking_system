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
