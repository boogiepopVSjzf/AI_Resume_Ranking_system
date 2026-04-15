from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.llm_score import ScoreOutput
from schemas.llm_explain import ExplanationOutput


class CandidateFinalResult(BaseModel):
    """
    Final downstream output for one candidate.
    This object matches the three-part output design:
    1. score
    2. raw pdf
    3. explanation
    """

    candidate_id: str = Field(..., description="Candidate identifier.")
    job_id: str = Field(..., description="Job identifier.")

    score: ScoreOutput = Field(..., description="Score-side output.")
    raw_pdf_path: str = Field(..., description="Original resume PDF reference.")
    explanation: ExplanationOutput = Field(..., description="Explanation-side output.")