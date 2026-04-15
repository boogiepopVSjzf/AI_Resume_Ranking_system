from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RuleType(str, Enum):
    """
    Supported rule types in the scoring system.
    """

    HARD_CONSTRAINT = "hard_constraint"
    SOFT = "soft"
    RED_FLAG = "red_flag"


class ScoringMode(str, Enum):
    """
    Supported scoring behaviors for one rule.
    """

    ADDITIVE = "additive"
    PENALTY = "penalty"
    CAP_SUBSCORE = "cap_subscore"
    GATE = "gate"


class RuleEvaluationResult(str, Enum):
    """
    Supported evaluation outcomes for one rule.
    """

    MET = "met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"
    UNKNOWN = "unknown"


class RecommendationDecision(str, Enum):
    """
    Final recommendation decision for one candidate.
    """

    RECOMMENDED = "recommended"
    RECOMMENDED_WITH_CAUTION = "recommended_with_caution"
    NOT_RECOMMENDED = "not_recommended"
    NEEDS_REVIEW = "needs_review"


class DimensionConfig(BaseModel):
    """
    Defines one score dimension in the active schema.
    Example: skills / experience / projects / education.
    """

    dimension_id: str = Field(..., description="Unique dimension identifier.")
    label: str = Field(..., description="Human-readable dimension label.")
    max_score: float = Field(..., ge=0, description="Maximum score for this dimension.")
    description: Optional[str] = Field(
        default=None,
        description="Optional explanation of this dimension.",
    )


class RuleActionConfig(BaseModel):
    """
    Defines how a rule affects scoring.
    """

    mode: ScoringMode = Field(..., description="Scoring behavior for this rule.")
    score_delta: Optional[float] = Field(
        default=None,
        description="Optional score delta for additive or penalty rules.",
    )
    max_score: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum allowed score for cap_subscore rules.",
    )
    decision_if_failed: Optional[RecommendationDecision] = Field(
        default=None,
        description="Decision to apply when a gate rule fails.",
    )


class EvidencePolicy(BaseModel):
    """
    Defines evidence requirements for rule evaluation.
    """

    min_evidence_items: int = Field(
        default=1,
        ge=0,
        description="Minimum number of evidence items required.",
    )
    allow_inferred_match: bool = Field(
        default=False,
        description="Whether inferred evidence is allowed.",
    )


class ScoringRule(BaseModel):
    """
    Defines one rule in the active scoring schema.
    """

    rule_id: str = Field(..., description="Unique rule identifier.")
    role_scope: str = Field(..., description="Role scope such as SWE / DS / PM.")
    rule_type: RuleType = Field(..., description="Type of the rule.")
    dimension: str = Field(..., description="Target dimension affected by this rule.")
    title: str = Field(..., description="Short rule title.")
    description: str = Field(..., description="Human-readable rule description.")
    keywords: List[str] = Field(
        default_factory=list,
        description="Optional keyword hints for downstream use.",
    )
    weight: float = Field(
        ...,
        ge=0,
        description="Relative importance of the rule.",
    )
    action: RuleActionConfig = Field(..., description="Scoring behavior for this rule.")
    evidence_policy: EvidencePolicy = Field(
        default_factory=EvidencePolicy,
        description="Evidence requirements for this rule.",
    )