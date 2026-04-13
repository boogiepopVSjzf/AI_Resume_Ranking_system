from __future__ import annotations

import json
from typing import Any, Optional

from config import settings
from schemas.job_query import HardFilters, VectorRetrieveResult
from services.resume_rag_split import (
    normalize_major,
    normalize_skill,
    parse_yoe_num,
)
from utils.errors import DatabaseError
from utils.logger import get_logger

logger = get_logger("postgres_store")

_SCHEMA_READY = False
_EDUCATION_RANK = {
    "high_school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}


def _extract_scalar(raw_json: dict[str, Any], field: str) -> Optional[str]:
    value = raw_json.get(field)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _extract_skills(raw_json: dict[str, Any]) -> list[str]:
    skills = raw_json.get("skills")
    if not isinstance(skills, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        if isinstance(skill, str):
            normalized = normalize_skill(skill)
            if normalized and normalized not in seen:
                seen.add(normalized)
                cleaned.append(normalized)
    return cleaned


def _extract_json_list(raw_json: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = raw_json.get(field)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _vector_literal(vector: Optional[list[float]]) -> Optional[str]:
    """Convert a Python vector into pgvector's text literal format."""
    if vector is None:
        return None
    if not vector:
        raise DatabaseError("Embedding vector is empty")
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _embedding_dimension(vector: Optional[list[float]]) -> int:
    if vector is not None:
        return len(vector)
    return settings.EMBEDDING_DIMENSION


def _connect():
    if not settings.DATABASE_URL:
        raise DatabaseError(
            "DATABASE_URL is not configured. Set it in your environment before enabling DB persistence."
        )

    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise DatabaseError(
            "psycopg is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    try:
        return psycopg.connect(
            settings.DATABASE_URL,
            autocommit=False,
            sslmode=settings.DATABASE_SSLMODE,
        )
    except Exception as exc:
        raise DatabaseError(f"Failed to connect to Postgres: {exc}") from exc


def _normalized_query_skills(skills: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        normalized = normalize_skill(skill)
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _build_hard_filter_query(hard_filters: HardFilters) -> tuple[str, dict[str, Any]]:
    where_clauses = ["1=1"]
    params: dict[str, Any] = {}

    if hard_filters.min_yoe is not None:
        where_clauses.append("yoe_num is not null and yoe_num >= %(min_yoe)s")
        params["min_yoe"] = hard_filters.min_yoe

    if hard_filters.major is not None:
        where_clauses.append("major = %(major)s")
        params["major"] = hard_filters.major

    if hard_filters.education_level is not None:
        if hard_filters.education_level == "other":
            where_clauses.append("education_level = %(education_level)s")
            params["education_level"] = hard_filters.education_level
        else:
            min_rank = _EDUCATION_RANK.get(hard_filters.education_level)
            if min_rank is not None:
                where_clauses.append(
                    """
                    (
                        case education_level
                            when 'high_school' then 1
                            when 'associate' then 2
                            when 'bachelor' then 3
                            when 'master' then 4
                            when 'phd' then 5
                            else 0
                        end
                    ) >= %(education_rank)s
                    """
                )
                params["education_rank"] = min_rank

    skills = _normalized_query_skills(hard_filters.required_skills)
    if skills:
        min_skill_matches = len(skills) if len(skills) < 3 else 3
        where_clauses.append(
            """
            (
                select count(*)
                from unnest(%(required_skills)s::text[]) as required_skill(skill)
                where required_skill.skill = any(coalesce(skills, '{}'::text[]))
            ) >= %(min_skill_matches)s
            """
        )
        params["required_skills"] = skills
        params["min_skill_matches"] = min_skill_matches

    query = f"""
        select resume_id
        from (
            select
                resume_id,
                row_number() over (
                    partition by
                        case
                            when coalesce(trim(email), '') <> '' and coalesce(trim(name), '') <> ''
                                then lower(trim(email)) || '|' || lower(trim(name))
                            else resume_id
                        end
                    order by created_at desc
                ) as rn,
                created_at
            from resumes
            where {' and '.join(where_clauses)}
        ) deduped
        where rn = 1
        order by created_at desc
    """
    return query, params


def _ensure_schema(conn, embedding_dimension: int) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY or not settings.DB_AUTO_INIT:
        return

    if embedding_dimension <= 0:
        raise DatabaseError("Embedding dimension must be a positive integer")

    statements = [
        "create extension if not exists vector",
        f"""
        create table if not exists resumes (
            resume_id text primary key,
            name text,
            email text,
            phone text,
            yoe_text text,
            yoe_num double precision,
            education_level text,
            major text,
            skills text[] not null default '{{}}',
            education_json jsonb not null default '[]'::jsonb,
            experience_json jsonb not null default '[]'::jsonb,
            projects_json jsonb not null default '[]'::jsonb,
            metadata jsonb not null,
            semantic_text text not null,
            raw_json jsonb not null,
            embedding vector({embedding_dimension}),
            source_file_name text,
            source_file_type text,
            pdf_storage_bucket text,
            pdf_storage_key text,
            pdf_mime_type text,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now()
        )
        """,
        "alter table resumes add column if not exists name text",
        "alter table resumes add column if not exists email text",
        "alter table resumes add column if not exists phone text",
        "alter table resumes add column if not exists yoe_text text",
        "alter table resumes add column if not exists yoe_num double precision",
        "alter table resumes add column if not exists education_level text",
        "alter table resumes add column if not exists major text",
        "alter table resumes add column if not exists skills text[] not null default '{}'",
        "alter table resumes add column if not exists education_json jsonb not null default '[]'::jsonb",
        "alter table resumes add column if not exists experience_json jsonb not null default '[]'::jsonb",
        "alter table resumes add column if not exists projects_json jsonb not null default '[]'::jsonb",
        "alter table resumes add column if not exists pdf_storage_bucket text",
        "alter table resumes add column if not exists pdf_storage_key text",
        "alter table resumes add column if not exists pdf_mime_type text",
        "alter table resumes drop column if exists summary",
        "alter table resumes drop column if exists txt_path",
        "drop index if exists idx_resumes_location",
        "alter table resumes drop column if exists location",
        "alter table resumes drop column if exists highest_education_level",
        "create index if not exists idx_resumes_email on resumes (email)",
        "create index if not exists idx_resumes_yoe_num on resumes (yoe_num)",
        "create index if not exists idx_resumes_education_level on resumes (education_level)",
        "create index if not exists idx_resumes_major on resumes (major)",
        "create index if not exists idx_resumes_skills_gin on resumes using gin (skills)",
        "create index if not exists idx_resumes_metadata_gin on resumes using gin (metadata)",
        "create index if not exists idx_resumes_raw_json_gin on resumes using gin (raw_json)",
        "create index if not exists idx_resumes_education_json_gin on resumes using gin (education_json)",
        "create index if not exists idx_resumes_experience_json_gin on resumes using gin (experience_json)",
        "create index if not exists idx_resumes_projects_json_gin on resumes using gin (projects_json)",
        "create index if not exists idx_resumes_embedding_hnsw on resumes using hnsw (embedding vector_cosine_ops)",
    ]

    try:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
        conn.commit()
        _SCHEMA_READY = True
    except Exception as exc:
        conn.rollback()
        raise DatabaseError(f"Failed to initialise Postgres schema: {exc}") from exc


def save_resume_bundle(
    *,
    resume_id: str,
    bundle: dict[str, Any],
    source_file_name: Optional[str] = None,
    source_file_type: Optional[str] = None,
    pdf_storage_bucket: Optional[str] = None,
    pdf_storage_key: Optional[str] = None,
    pdf_mime_type: Optional[str] = None,
) -> None:
    """Upsert one parsed resume bundle into Postgres."""
    metadata = bundle.get("metadata", {})
    semantic_text = bundle.get("semantic_text") or ""
    raw_json = bundle.get("raw_json", {})
    if not isinstance(raw_json, dict):
        raise DatabaseError("raw_json must be a JSON object")
    embedding = bundle.get("embedding")
    name = _extract_scalar(raw_json, "name")
    email = _extract_scalar(raw_json, "email")
    phone = _extract_scalar(raw_json, "phone")
    yoe_text = _extract_scalar(raw_json, "YoE")
    yoe_num = parse_yoe_num(yoe_text)
    education_level = _extract_scalar(raw_json, "education_level")
    skills = _extract_skills(raw_json)
    education_json = _extract_json_list(raw_json, "education")
    experience_json = _extract_json_list(raw_json, "experience")
    projects_json = _extract_json_list(raw_json, "projects")
    major = _extract_scalar(raw_json, "major")
    if major is None:
        for education_item in education_json:
            raw_major = education_item.get("major")
            if isinstance(raw_major, str):
                major = normalize_major(raw_major)
                if major:
                    break

    embedding_dimension = _embedding_dimension(embedding)

    with _connect() as conn:
        _ensure_schema(conn, embedding_dimension)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into resumes (
                        resume_id,
                        name,
                        email,
                        phone,
                        yoe_text,
                        yoe_num,
                        education_level,
                        major,
                        skills,
                        education_json,
                        experience_json,
                        projects_json,
                        metadata,
                        semantic_text,
                        raw_json,
                        embedding,
                        source_file_name,
                        source_file_type,
                        pdf_storage_bucket,
                        pdf_storage_key,
                        pdf_mime_type
                    )
                    values (
                        %(resume_id)s,
                        %(name)s,
                        %(email)s,
                        %(phone)s,
                        %(yoe_text)s,
                        %(yoe_num)s,
                        %(education_level)s,
                        %(major)s,
                        %(skills)s,
                        %(education_json)s::jsonb,
                        %(experience_json)s::jsonb,
                        %(projects_json)s::jsonb,
                        %(metadata)s::jsonb,
                        %(semantic_text)s,
                        %(raw_json)s::jsonb,
                        %(embedding)s::vector,
                        %(source_file_name)s,
                        %(source_file_type)s,
                        %(pdf_storage_bucket)s,
                        %(pdf_storage_key)s,
                        %(pdf_mime_type)s
                    )
                    on conflict (resume_id) do update
                    set
                        name = excluded.name,
                        email = excluded.email,
                        phone = excluded.phone,
                        yoe_text = excluded.yoe_text,
                        yoe_num = excluded.yoe_num,
                        education_level = excluded.education_level,
                        major = excluded.major,
                        skills = excluded.skills,
                        education_json = excluded.education_json,
                        experience_json = excluded.experience_json,
                        projects_json = excluded.projects_json,
                        metadata = excluded.metadata,
                        semantic_text = excluded.semantic_text,
                        raw_json = excluded.raw_json,
                        embedding = excluded.embedding,
                        source_file_name = excluded.source_file_name,
                        source_file_type = excluded.source_file_type,
                        pdf_storage_bucket = excluded.pdf_storage_bucket,
                        pdf_storage_key = excluded.pdf_storage_key,
                        pdf_mime_type = excluded.pdf_mime_type,
                        updated_at = now()
                    """,
                    {
                        "resume_id": resume_id,
                        "name": name,
                        "email": email,
                        "phone": phone,
                        "yoe_text": yoe_text,
                        "yoe_num": yoe_num,
                        "education_level": education_level,
                        "major": major,
                        "skills": skills,
                        "education_json": json.dumps(education_json, ensure_ascii=False),
                        "experience_json": json.dumps(experience_json, ensure_ascii=False),
                        "projects_json": json.dumps(projects_json, ensure_ascii=False),
                        "metadata": json.dumps(metadata, ensure_ascii=False),
                        "semantic_text": semantic_text,
                        "raw_json": json.dumps(raw_json, ensure_ascii=False),
                        "embedding": _vector_literal(embedding),
                        "source_file_name": source_file_name,
                        "source_file_type": source_file_type,
                        "pdf_storage_bucket": pdf_storage_bucket,
                        "pdf_storage_key": pdf_storage_key,
                        "pdf_mime_type": pdf_mime_type,
                    },
                )
            conn.commit()
            logger.info("Persisted resume %s to Postgres", resume_id)
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to save resume bundle: {exc}") from exc


def query_resume_ids_by_hard_filters(hard_filters: HardFilters) -> list[str]:
    """Return resume IDs that satisfy the first-stage SQL hard filters."""
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        query, params = _build_hard_filter_query(hard_filters)
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return [row[0] for row in rows]
        except Exception as exc:
            raise DatabaseError(f"Failed to query resume ids by hard filters: {exc}") from exc


def query_similar_resumes(
    *,
    resume_ids: list[str],
    search_query_embedding: list[float],
    top_k: int = 10,
) -> list[VectorRetrieveResult]:
    """Rank candidate resumes by vector similarity within a filtered candidate pool."""
    if not resume_ids:
        return []
    if not search_query_embedding:
        raise DatabaseError("search_query_embedding cannot be empty")
    if top_k <= 0:
        raise DatabaseError("top_k must be a positive integer")

    query = """
        select
            resume_id,
            1 - (embedding <=> %(query_embedding)s::vector) as similarity_score,
            metadata,
            raw_json
        from resumes
        where resume_id = any(%(resume_ids)s::text[])
          and embedding is not null
        order by embedding <=> %(query_embedding)s::vector
        limit %(top_k)s
    """

    params = {
        "query_embedding": _vector_literal(search_query_embedding),
        "resume_ids": resume_ids,
        "top_k": top_k,
    }

    with _connect() as conn:
        _ensure_schema(conn, len(search_query_embedding))
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return [
                VectorRetrieveResult(
                    resume_id=row[0],
                    similarity_score=float(row[1]) if row[1] is not None else 0.0,
                    metadata=row[2] if isinstance(row[2], dict) else {},
                    raw_json=row[3] if isinstance(row[3], dict) else {},
                )
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to query similar resumes: {exc}") from exc
