# Cloud DB Plan

## 1. Problem

The current local SQLite database is sufficient for individual development, but it cannot support team-wide integration testing because other team members cannot access the same local database instance.

## 2. Goal

We need a shared database solution so that the team can:
- access the same stored resume data,
- reproduce end-to-end tests,
- support the full pipeline from parsing to scoring.

## 3. Current Status

At the current stage, the parsing pipeline is:

resume -> txt -> structured json -> database

This local persistence is useful for prototyping, but it is not enough for shared team testing.

## 4. Requirements for Shared Database

The shared database should support:
- common team access,
- stable read/write behavior,
- storage of parsed resume profiles,
- future storage of good-case resumes for retrieval,
- support for end-to-end pipeline testing.

## 5. Proposed Architecture

### Local development
- Use SQLite for fast local prototyping.
- Keep the current repository interface stable.

### Shared team testing
- Use a shared cloud-hosted relational database.
- Access should be controlled through common team credentials or environment variables.

## 6. Interface Design

The storage layer should keep the same high-level interface regardless of backend:

- `get_connection()`
- `init_db()`
- `save_parsed_resume(resume_id, structured)`
- `save_good_case(case_payload)`
- `retrieve_similar_cases(query_payload, top_k)`

This design allows migration from local SQLite to a shared cloud database with minimal changes in upper-layer services.
