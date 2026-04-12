from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

EducationLevel = Literal["high_school", "associate", "bachelor", "master", "phd", "other"]
MajorCategory = Literal[
    "computer_science",
    "mathematics",
    "medicine",
    "finance",
    "engineering",
    "other",
]

#定义输出json结构，包括教育经历、工作经历、项目经历、技能、摘要、姓名、邮箱、手机号、工作经历、最高学历、位置
class EducationItem(BaseModel):
    school: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class ExperienceItem(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class ResumeStructured(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    YoE: Optional[str] = None
    education_level: Optional[EducationLevel] = None
    major: Optional[MajorCategory] = None
    location: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)

class ExtractionInput(BaseModel):
    resume_id: Optional[str] = None
    text: str
