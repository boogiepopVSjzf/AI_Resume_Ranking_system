from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from schemas.score_shared import RuleEvaluationResult


class RuleExplanationItem(BaseModel):
    """
    User-facing explanation item corresponding to one evaluated rule.
    """

    rule_id: str = Field(..., description="Rule identifier.")
    title: str = Field(..., description="Rule title.")
    result: RuleEvaluationResult = Field(..., description="Rule evaluation result.")
    impact: Optional[str] = Field(
        default=None,
        description="Human-readable description of rule impact.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Human-readable evidence items.",
    )


class ExplanationSchemaConfig(BaseModel):
    """
    Optional configuration document for explanation generation.
    This can be employer-selected independently from the score schema if needed.
    """

    schema_id: str = Field(..., description="Unique explanation schema identifier.")
    schema_name: str = Field(..., description="Human-readable explanation schema name.")
    version: str = Field(..., description="Explanation schema version.")
    require_summary: bool = True
    require_strengths: bool = True
    require_gaps: bool = True
    require_red_flags: bool = True
    require_rule_based_analysis: bool = True
    require_final_recommendation_reason: bool = True


class ExplanationOutput(BaseModel):
    """
    Final explanation-side output for one candidate.
    This object should only contain explanation-related fields.
    """

    candidate_id: str = Field(..., description="Candidate identifier.")
    job_id: str = Field(..., description="Job identifier.")
    summary: str = Field(..., description="Short overall summary of candidate fit.")

    strengths: List[str] = Field(
        default_factory=list,
        description="Main strengths identified for the candidate.",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Main gaps or missing requirements.",
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Detected red flags.",
    )

    rule_based_analysis: List[RuleExplanationItem] = Field(
        default_factory=list,
        description="Rule-by-rule explanation summary.",
    )
