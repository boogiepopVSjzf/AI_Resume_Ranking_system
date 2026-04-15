from __future__ import annotations

from schemas.llm_explain import ExplanationSchemaConfig
from schemas.llm_score import OutputPolicy, ScoreSchema
from schemas.score_shared import (
    DimensionConfig,
    EvidencePolicy,
    RecommendationDecision,
    RuleActionConfig,
    RuleType,
    ScoringMode,
    ScoringRule,
)


SCORE_SCHEMA = ScoreSchema(
    schema_id="software_engineer_v1",
    schema_name="Software Engineer Scoring Schema",
    role_scope="software_engineer",
    version="1.0",
    description="Preset scoring schema for software engineer roles.",
    total_score=100.0,
    dimensions=[
        DimensionConfig(
            dimension_id="cs_fundamentals",
            label="CS Fundamentals",
            max_score=20.0,
            description="Algorithms, data structures, and systems foundation.",
        ),
        DimensionConfig(
            dimension_id="coding_skills",
            label="Coding Skills",
            max_score=30.0,
            description="Programming language and implementation ability.",
        ),
        DimensionConfig(
            dimension_id="engineering_experience",
            label="Engineering Experience",
            max_score=30.0,
            description="Relevant software engineering project or work experience.",
        ),
        DimensionConfig(
            dimension_id="system_design",
            label="System Design",
            max_score=10.0,
            description="Architecture and scalability awareness.",
        ),
        DimensionConfig(
            dimension_id="collaboration",
            label="Collaboration",
            max_score=10.0,
            description="Teamwork and communication fit.",
        ),
    ],
    rules=[
        ScoringRule(
            rule_id="se_programming_required",
            role_scope="software_engineer",
            rule_type=RuleType.HARD_CONSTRAINT,
            dimension="coding_skills",
            title="Core Programming Ability",
            description="Candidate should show clear programming ability in one or more mainstream languages.",
            keywords=["python", "java", "c++", "javascript", "go"],
            weight=1.0,
            action=RuleActionConfig(
                mode=ScoringMode.GATE,
                decision_if_failed=RecommendationDecision.NOT_RECOMMENDED,
            ),
            evidence_policy=EvidencePolicy(
                min_evidence_items=1,
                allow_inferred_match=False,
            ),
        ),
        ScoringRule(
            rule_id="se_backend_fullstack_bonus",
            role_scope="software_engineer",
            rule_type=RuleType.SOFT,
            dimension="engineering_experience",
            title="Backend or Full-stack Experience",
            description="Candidate is preferred if they have backend or full-stack engineering experience.",
            keywords=["backend", "api", "database", "full stack", "microservice"],
            weight=1.0,
            action=RuleActionConfig(
                mode=ScoringMode.ADDITIVE,
                score_delta=15.0,
            ),
            evidence_policy=EvidencePolicy(
                min_evidence_items=1,
                allow_inferred_match=True,
            ),
        ),
        ScoringRule(
            rule_id="se_system_design_bonus",
            role_scope="software_engineer",
            rule_type=RuleType.SOFT,
            dimension="system_design",
            title="System Design Awareness",
            description="Candidate is preferred if they show architecture, distributed systems, or scalability awareness.",
            keywords=["architecture", "distributed systems", "scalability", "design"],
            weight=0.8,
            action=RuleActionConfig(
                mode=ScoringMode.ADDITIVE,
                score_delta=8.0,
            ),
            evidence_policy=EvidencePolicy(
                min_evidence_items=1,
                allow_inferred_match=True,
            ),
        ),
        ScoringRule(
            rule_id="se_collaboration_bonus",
            role_scope="software_engineer",
            rule_type=RuleType.SOFT,
            dimension="collaboration",
            title="Team Collaboration",
            description="Candidate is preferred if they demonstrate teamwork, code review, or cross-functional collaboration.",
            keywords=["collaboration", "team", "code review", "cross-functional"],
            weight=0.7,
            action=RuleActionConfig(
                mode=ScoringMode.ADDITIVE,
                score_delta=7.0,
            ),
            evidence_policy=EvidencePolicy(
                min_evidence_items=1,
                allow_inferred_match=True,
            ),
        ),
        ScoringRule(
            rule_id="se_low_fit_red_flag",
            role_scope="software_engineer",
            rule_type=RuleType.RED_FLAG,
            dimension="engineering_experience",
            title="Weak Software Engineering Fit",
            description="Trigger a penalty if the profile appears broadly unrelated to software engineering execution.",
            keywords=["unrelated"],
            weight=0.8,
            action=RuleActionConfig(
                mode=ScoringMode.PENALTY,
                score_delta=10.0,
            ),
            evidence_policy=EvidencePolicy(
                min_evidence_items=0,
                allow_inferred_match=True,
            ),
        ),
    ],
    output_policy=OutputPolicy(
        require_score_breakdown=True,
        require_rule_results=True,
        require_dimension_reasons=True,
    ),
)


EXPLANATION_SCHEMA = ExplanationSchemaConfig(
    schema_id="software_engineer_v1_explanation",
    schema_name="Software Engineer Explanation Schema",
    version="1.0",
    require_summary=True,
    require_strengths=True,
    require_gaps=True,
    require_red_flags=True,
    require_rule_based_analysis=True,
    require_final_recommendation_reason=True,
)