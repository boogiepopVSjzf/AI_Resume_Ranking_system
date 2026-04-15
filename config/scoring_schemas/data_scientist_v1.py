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
    schema_id="data_scientist_v1",
    schema_name="Data Scientist Scoring Schema",
    role_scope="data_scientist",
    version="1.0",
    description="Preset scoring schema for data scientist roles.",
    total_score=100.0,
    dimensions=[
        DimensionConfig(
            dimension_id="education",
            label="Education",
            max_score=15.0,
            description="Education fit for the role.",
        ),
        DimensionConfig(
            dimension_id="technical_skills",
            label="Technical Skills",
            max_score=35.0,
            description="Programming, data, and machine learning skill fit.",
        ),
        DimensionConfig(
            dimension_id="experience",
            label="Experience",
            max_score=30.0,
            description="Relevant work, research, or project experience.",
        ),
        DimensionConfig(
            dimension_id="communication",
            label="Communication",
            max_score=10.0,
            description="Communication and stakeholder-facing capability.",
        ),
        DimensionConfig(
            dimension_id="role_alignment",
            label="Role Alignment",
            max_score=10.0,
            description="Overall alignment with the JD.",
        ),
    ],
    rules=[
        ScoringRule(
            rule_id="ds_python_sql_required",
            role_scope="data_scientist",
            rule_type=RuleType.HARD_CONSTRAINT,
            dimension="technical_skills",
            title="Python and SQL Baseline",
            description="Candidate should show evidence of both Python and SQL.",
            keywords=["python", "sql"],
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
            rule_id="ds_ml_stats_bonus",
            role_scope="data_scientist",
            rule_type=RuleType.SOFT,
            dimension="technical_skills",
            title="Machine Learning and Statistics",
            description="Candidate is preferred if resume shows machine learning and statistics background.",
            keywords=["machine learning", "statistics", "modeling", "regression"],
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
            rule_id="ds_data_project_bonus",
            role_scope="data_scientist",
            rule_type=RuleType.SOFT,
            dimension="experience",
            title="Relevant Data Project Experience",
            description="Candidate is preferred if they show data analysis, ML, NLP, or analytics project experience.",
            keywords=["project", "analysis", "nlp", "research", "analytics"],
            weight=0.9,
            action=RuleActionConfig(
                mode=ScoringMode.ADDITIVE,
                score_delta=12.0,
            ),
            evidence_policy=EvidencePolicy(
                min_evidence_items=1,
                allow_inferred_match=True,
            ),
        ),
        ScoringRule(
            rule_id="ds_communication_bonus",
            role_scope="data_scientist",
            rule_type=RuleType.SOFT,
            dimension="communication",
            title="Communication of Insights",
            description="Candidate is preferred if they can explain insights, experiments, or results clearly.",
            keywords=["presentation", "stakeholder", "report", "insight", "visualization"],
            weight=0.7,
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
            rule_id="ds_unrelated_profile_red_flag",
            role_scope="data_scientist",
            rule_type=RuleType.RED_FLAG,
            dimension="role_alignment",
            title="Weak Overall Data Science Alignment",
            description="Trigger a penalty if the profile appears broadly unrelated to data science work.",
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
    schema_id="data_scientist_v1_explanation",
    schema_name="Data Scientist Explanation Schema",
    version="1.0",
    require_summary=True,
    require_strengths=True,
    require_gaps=True,
    require_red_flags=True,
    require_rule_based_analysis=True,
    require_final_recommendation_reason=True,
)