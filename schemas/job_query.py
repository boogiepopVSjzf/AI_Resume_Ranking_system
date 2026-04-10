from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class HardFilters(BaseModel):
    min_yoe: Optional[int] = None
    required_skills: List[str] = []
    education: Optional[str] = None
    location: Optional[str] = None


class StandardizedJobQuery(BaseModel):
    hard_filters: HardFilters
    search_query: str
