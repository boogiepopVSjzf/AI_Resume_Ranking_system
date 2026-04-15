from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from schemas.llm_explain import ExplanationSchemaConfig
from schemas.llm_score import ScoreSchema

from config.scoring_schemas.data_scientist_v1 import (
    EXPLANATION_SCHEMA as DATA_SCIENTIST_EXPLANATION_SCHEMA,
    SCORE_SCHEMA as DATA_SCIENTIST_SCORE_SCHEMA,
)
from config.scoring_schemas.software_engineer_v1 import (
    EXPLANATION_SCHEMA as SOFTWARE_ENGINEER_EXPLANATION_SCHEMA,
    SCORE_SCHEMA as SOFTWARE_ENGINEER_SCORE_SCHEMA,
)


@dataclass(frozen=True)
class ScoringSchemaBundle:
    """
    Internal bundle used by the scoring service.
    """

    schema_key: str
    score_schema: ScoreSchema
    explanation_schema: ExplanationSchemaConfig


_SCHEMA_REGISTRY: Dict[str, ScoringSchemaBundle] = {
    "data_scientist_v1": ScoringSchemaBundle(
        schema_key="data_scientist_v1",
        score_schema=DATA_SCIENTIST_SCORE_SCHEMA,
        explanation_schema=DATA_SCIENTIST_EXPLANATION_SCHEMA,
    ),
    "software_engineer_v1": ScoringSchemaBundle(
        schema_key="software_engineer_v1",
        score_schema=SOFTWARE_ENGINEER_SCORE_SCHEMA,
        explanation_schema=SOFTWARE_ENGINEER_EXPLANATION_SCHEMA,
    ),
}


def get_schema_bundle(schema_key: str) -> ScoringSchemaBundle:
    """
    Return one preset schema bundle by key.
    """
    bundle = _SCHEMA_REGISTRY.get(schema_key)
    if bundle is None:
        supported = ", ".join(sorted(_SCHEMA_REGISTRY.keys()))
        raise ValueError(
            f"Unknown schema_key '{schema_key}'. Supported values: {supported}"
        )
    return bundle


def list_schema_keys() -> List[str]:
    """
    Return all supported schema keys.
    """
    return sorted(_SCHEMA_REGISTRY.keys())