from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from schemas.score_shared import (
    DimensionConfig,
    RecommendationDecision,
    RuleEvaluationResult,
    RuleType,
    ScoringMode,
    ScoringRule,
)

class ResumeJsonInput(BaseModel):
    """
    Generic resume JSON payload passed from hard-filter / vector-retrieval stage.
    Keep this flexible to avoid breaking the existing database shape.
    """
    candidate_id: Optional[str] = Field(default=None)
    raw_pdf_path: Optional[str] = Field(default=None)
    data: Dict[str, Any] = Field(default_factory=dict)


class JDJsonInput(BaseModel):
    """
    Generic JD JSON payload passed from job-context / query-rewrite stage.
    Keep this flexible to preserve the existing upstream shape.
    """
    job_id: Optional[str] = Field(default=None)
    data: Dict[str, Any] = Field(default_factory=dict)


class LLMScoringBatchRequest(BaseModel):
    """
    Batch scoring request:
    - one or more resume JSON objects
    - one JD JSON object
    - score/explanation schemas
    """
    resumes: List[Dict[str, Any]] = Field(..., min_length=1)
    jd: Dict[str, Any] = Field(...)
    score_schema: Dict[str, Any] = Field(...)
    explanation_schema: Dict[str, Any] = Field(...)
    provider: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)

class OutputPolicy(BaseModel):
    """
    Defines score-output requirements for an active schema.
    """

    require_score_breakdown: bool = True
    require_rule_results: bool = True
    require_dimension_reasons: bool = True


class ScoreSchema(BaseModel):
    """
    Employer-selected active scoring schema for one role.
    This document only focuses on score generation.
    """

    schema_id: str = Field(..., description="Unique schema identifier.")
    schema_name: str = Field(..., description="Human-readable schema name.")
    role_scope: str = Field(..., description="Role scope such as SWE / DS / PM.")
    version: str = Field(..., description="Schema version string.")
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the score schema.",
    )

    total_score: float = Field(..., gt=0, description="Maximum overall score.")
    dimensions: List[DimensionConfig] = Field(
        default_factory=list,
        description="Configured score dimensions.",
    )
    rules: List[ScoringRule] = Field(
        default_factory=list,
        description="All active scoring rules.",
    )
    output_policy: OutputPolicy = Field(
        default_factory=OutputPolicy,
        description="Output requirements for score generation.",
    )

    @model_validator(mode="after")
    def validate_dimensions_and_rules(self) -> "ScoreSchema":
        """
        Validate that all rules reference known dimensions.
        """
        dimension_ids = {dimension.dimension_id for dimension in self.dimensions}

        for rule in self.rules:
            if rule.dimension not in dimension_ids:
                raise ValueError(
                    f"Rule '{rule.rule_id}' references unknown dimension '{rule.dimension}'."
                )

        return self


class ScoreImpact(BaseModel):
    """
    Structured record of how one rule changed the score.
    """

    type: ScoringMode = Field(..., description="Impact type.")
    value: Optional[float] = Field(
        default=None,
        description="Numeric impact value if applicable.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional short explanation of the impact.",
    )


class RuleScoreResult(BaseModel):
    """
    Score-side evaluation result for one rule.
    """

    rule_id: str = Field(..., description="Rule identifier.")
    rule_type: RuleType = Field(..., description="Rule type.")
    dimension: str = Field(..., description="Target dimension.")
    result: RuleEvaluationResult = Field(..., description="Rule evaluation result.")
    applied: bool = Field(..., description="Whether this rule affected the score.")
    score_impact: Optional[ScoreImpact] = Field(
        default=None,
        description="How the rule affected the score.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the rule result.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional note for debugging or auditing.",
    )


class DimensionScore(BaseModel):
    """
    Final score for one dimension.
    """

    dimension_id: str = Field(..., description="Dimension identifier.")
    score: float = Field(..., ge=0, description="Assigned score.")
    max_score: float = Field(..., ge=0, description="Maximum allowed score.")
    reason: Optional[str] = Field(
        default=None,
        description="Short explanation for this dimension score.",
    )

    @model_validator(mode="after")
    def validate_score_not_exceed_max(self) -> "DimensionScore":
        """
        Validate that score does not exceed max_score.
        """
        if self.score > self.max_score:
            raise ValueError(
                f"Dimension '{self.dimension_id}' score ({self.score}) "
                f"cannot exceed max_score ({self.max_score})."
            )
        return self


class ScoreOutput(BaseModel):
    """
    Final score-side output for one candidate.
    This object should only contain score-related fields.
    """

    candidate_id: str = Field(..., description="Candidate identifier.")
    job_id: str = Field(..., description="Job identifier.")
    schema_id: str = Field(..., description="Active score schema identifier.")

    final_score: float = Field(..., ge=0, description="Final candidate score.")
    decision: RecommendationDecision = Field(
        ...,
        description="Final recommendation decision.",
    )

    dimension_scores: List[DimensionScore] = Field(
        default_factory=list,
        description="Per-dimension scores.",
    )
    rule_results: List[RuleScoreResult] = Field(
        default_factory=list,
        description="Detailed rule-level score results.",
    )

    llm_usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional token usage or provider metadata.",
    )

    @model_validator(mode="after")
    def validate_final_score(self) -> "ScoreOutput":
        """
        Lightweight consistency check between final score and dimension scores.
        """
        if self.dimension_scores:
            dimension_sum = sum(item.score for item in self.dimension_scores)
            if self.final_score > dimension_sum + 1e-6:
                raise ValueError(
                    f"final_score ({self.final_score}) cannot exceed the sum of "
                    f"dimension_scores ({dimension_sum})."
                )
        return self