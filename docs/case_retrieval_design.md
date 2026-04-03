# Case Retrieval Design

## 1. Goal

The goal of case retrieval is to store good resumes as case memory and retrieve the most similar historical good cases for a new input resume.

This module will support the later scoring stage by providing relevant good-case evidence to the ranking/scoring pipeline.

## 2. High-Level Logic

The intended high-level flow is:

1. Parse incoming resume into structured schema.
2. Build a query using:
   - schema summary,
   - job description summary,
   - company criteria,
   - role title.
3. Retrieve top-k similar good cases from the database.
4. Package retrieved cases together with current resume evidence for downstream scoring.

## 3. Inputs

The retrieval module should accept:

- `schema_summary: dict`
- `job_description_summary: str`
- `criteria: dict`
- `role_title: str`
- `top_k: int`

## 4. Outputs

The retrieval module should return:

- a list of top-k similar good cases,
- optional retrieval metadata such as similarity score,
- identifiers for the matched cases.

## 5. Proposed Files

### `storage/case_repository.py`
Responsible for database read/write operations for case memory.

### `services/case_retrieval_service.py`
Responsible for retrieval workflow and orchestration logic.

### `schemas/case_models.py`
Responsible for request/response schemas for case retrieval.

## 6. Proposed Functions

### `save_good_case(case_id: str, case_payload: dict) -> None`
Store a good resume case into the database after the scoring stage confirms it is a good case.

### `build_case_query(schema_summary: dict, job_description_summary: str, criteria: dict, role_title: str) -> dict`
Construct a normalized query payload for retrieval.

### `retrieve_similar_cases(query_payload: dict, top_k: int) -> list[dict]`
Retrieve the top-k most similar good cases from case memory.

### `list_good_cases_by_role(role_title: str, limit: int) -> list[dict]`
List stored good cases for a given role.

### `format_cases_for_scoring(cases: list[dict]) -> dict`
Prepare retrieved cases for downstream scoring or LLM packaging.

## 7. Data to Store for Each Good Case

A stored good case may include:

- case id,
- resume id,
- role title,
- parsed profile/schema,
- total score,
- section scores,
- similarity to job description,
- evidence,
- missing skills,
- reason,
- model metadata,
- created timestamp.

## 8. Dependencies

This design depends on:
- the parsing schema,
- the scoring output schema,
- agreement with Claire on the final good-resume output structure.
