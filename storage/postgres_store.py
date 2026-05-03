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
        """
        create index if not exists idx_resumes_email_lower
        on resumes (lower(trim(email)))
        where email is not null and trim(email) <> ''
        """,
        """
        create index if not exists idx_resumes_name_lower_when_email_missing
        on resumes (lower(trim(name)))
        where (email is null or trim(email) = '')
          and name is not null and trim(name) <> ''
        """,
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
        f"""
        create table if not exists scoring_schemas (
            schema_id text primary key,
            schema_name text not null,
            rules_json jsonb not null,
            summary text not null,
            embedding vector({embedding_dimension}),
            version integer not null default 1,
            is_active boolean not null default true,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now(),
            unique (schema_name, version)
        )
        """,
        "create index if not exists idx_scoring_schemas_name on scoring_schemas (schema_name)",
        "create index if not exists idx_scoring_schemas_active on scoring_schemas (is_active)",
        "create index if not exists idx_scoring_schemas_embedding_hnsw on scoring_schemas using hnsw (embedding vector_cosine_ops)",
        """
        create table if not exists feedback_examples (
            feedback_id text primary key,
            schema_id text not null references scoring_schemas(schema_id) on delete cascade,
            resume_id text not null references resumes(resume_id) on delete cascade,
            label text not null check (label in ('excellent', 'good', 'qualified', 'bad', 'n/a')),
            feedback_text text,
            score double precision check (score is null or (score >= 0 and score <= 10)),
            scoring_result jsonb,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now()
        )
        """,
        "alter table feedback_examples drop constraint if exists feedback_examples_label_check",
        """
        alter table feedback_examples
        add constraint feedback_examples_label_check
        check (label in ('excellent', 'good', 'qualified', 'bad', 'n/a'))
        """,
        "create index if not exists idx_feedback_examples_schema_id on feedback_examples (schema_id)",
        "create index if not exists idx_feedback_examples_resume_id on feedback_examples (resume_id)",
        "create index if not exists idx_feedback_examples_label on feedback_examples (label)",
        "create index if not exists idx_feedback_examples_schema_label on feedback_examples (schema_id, label)",
        # ── Chat session tables (agent multi-turn conversation) ──────────────
        """
        create table if not exists chat_sessions (
            session_id       text primary key,
            title            text,
            active_schema_id text references scoring_schemas(schema_id) on delete set null,
            hr_auto_mode     boolean not null default false,
            provider         text,
            model            text,
            created_at       timestamptz not null default now(),
            updated_at       timestamptz not null default now()
        )
        """,
        "create index if not exists idx_chat_sessions_created_at on chat_sessions (created_at desc)",
        """
        create table if not exists chat_messages (
            message_id      text primary key,
            session_id      text not null references chat_sessions(session_id) on delete cascade,
            role            text not null check (role in ('user','assistant','tool','system')),
            content         text not null,
            tool_name       text,
            tool_call_id    text,
            tool_calls_json jsonb,
            created_at      timestamptz not null default now()
        )
        """,
        "create index if not exists idx_chat_messages_session on chat_messages (session_id, created_at)",
        # ── Long-term memory table ───────────────────────────────────────────
        f"""
        create table if not exists agent_memories (
            memory_id       text primary key,
            session_id      text references chat_sessions(session_id) on delete set null,
            memory_type     text not null check (memory_type in (
                                'hr_preference', 'session_summary',
                                'search_result_cache', 'schema_draft_history')),
            content         text not null,
            content_json    jsonb,
            embedding       vector({embedding_dimension}),
            expires_at      timestamptz,
            created_at      timestamptz not null default now(),
            updated_at      timestamptz not null default now()
        )
        """,
        "create index if not exists idx_agent_memories_session on agent_memories (session_id)",
        "create index if not exists idx_agent_memories_type on agent_memories (memory_type)",
        "create index if not exists idx_agent_memories_expires on agent_memories (expires_at) where expires_at is not null",
        "create index if not exists idx_agent_memories_embedding_hnsw on agent_memories using hnsw (embedding vector_cosine_ops) where embedding is not null",
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
                original_resume_id = resume_id
                if email:
                    cur.execute(
                        """
                        select resume_id
                        from resumes
                        where lower(trim(email)) = lower(trim(%(email)s))
                        order by updated_at desc, created_at desc
                        limit 1
                        """,
                        {"email": email},
                    )
                    row = cur.fetchone()
                    if row is not None:
                        resume_id = row[0]
                elif name:
                    cur.execute(
                        """
                        select resume_id
                        from resumes
                        where (email is null or trim(email) = '')
                          and lower(trim(name)) = lower(trim(%(name)s))
                        order by updated_at desc, created_at desc
                        limit 1
                        """,
                        {"name": name},
                    )
                    row = cur.fetchone()
                    if row is not None:
                        resume_id = row[0]

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
            if resume_id != original_resume_id:
                logger.info(
                    "Updated existing resume %s for duplicate candidate; discarded new resume_id %s",
                    resume_id,
                    original_resume_id,
                )
            else:
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
        order by embedding <=> %(query_embedding)s::vector, resume_id
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


def save_scoring_schema(
    *,
    schema_id: str,
    schema_name: str,
    rules_json: dict[str, Any],
    summary: str,
    embedding: Optional[list[float]],
) -> dict[str, Any]:
    """Create a new scoring schema version and persist its rules, summary, and embedding."""
    if not schema_id:
        raise DatabaseError("schema_id cannot be empty")
    if not schema_name.strip():
        raise DatabaseError("schema_name cannot be empty")
    if not summary.strip():
        raise DatabaseError("summary cannot be empty")

    embedding_dimension = _embedding_dimension(embedding)

    with _connect() as conn:
        _ensure_schema(conn, embedding_dimension)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select coalesce(max(version), 0) + 1
                    from scoring_schemas
                    where schema_name = %(schema_name)s
                    """,
                    {"schema_name": schema_name},
                )
                version = cur.fetchone()[0]
                cur.execute(
                    """
                    update scoring_schemas
                    set is_active = false,
                        updated_at = now()
                    where schema_name = %(schema_name)s
                      and is_active = true
                    """,
                    {"schema_name": schema_name},
                )
                cur.execute(
                    """
                    insert into scoring_schemas (
                        schema_id,
                        schema_name,
                        rules_json,
                        summary,
                        embedding,
                        version,
                        is_active
                    )
                    values (
                        %(schema_id)s,
                        %(schema_name)s,
                        %(rules_json)s::jsonb,
                        %(summary)s,
                        %(embedding)s::vector,
                        %(version)s,
                        true
                    )
                    """,
                    {
                        "schema_id": schema_id,
                        "schema_name": schema_name,
                        "rules_json": json.dumps(rules_json, ensure_ascii=False),
                        "summary": summary,
                        "embedding": _vector_literal(embedding),
                        "version": version,
                    },
                )
            conn.commit()
            logger.info("Persisted scoring schema %s v%s", schema_name, version)
            return {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "rules_json": rules_json,
                "summary": summary,
                "version": version,
                "is_active": True,
            }
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to save scoring schema: {exc}") from exc


def find_best_scoring_schema(query_embedding: list[float]) -> dict[str, Any]:
    """Find the active scoring schema closest to the JD/query embedding."""
    if not query_embedding:
        raise DatabaseError("query_embedding cannot be empty")

    query = """
        select
            schema_id,
            schema_name,
            rules_json,
            summary,
            version,
            1 - (embedding <=> %(query_embedding)s::vector) as similarity_score
        from scoring_schemas
        where is_active = true
          and embedding is not null
        order by embedding <=> %(query_embedding)s::vector, schema_name, version desc, schema_id
        limit 1
    """

    with _connect() as conn:
        _ensure_schema(conn, len(query_embedding))
        try:
            with conn.cursor() as cur:
                cur.execute(query, {"query_embedding": _vector_literal(query_embedding)})
                row = cur.fetchone()
            if row is None:
                raise DatabaseError("No active scoring schema with embedding was found")
            return {
                "schema_id": row[0],
                "schema_name": row[1],
                "rules_json": row[2] if isinstance(row[2], dict) else {},
                "summary": row[3],
                "version": row[4],
                "similarity_score": float(row[5]) if row[5] is not None else 0.0,
            }
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Failed to find best scoring schema: {exc}") from exc


def get_feedback_examples(
    *,
    schema_id: str,
    limit_per_label: int = 2,
) -> list[dict[str, Any]]:
    """Return recent feedback examples for a schema, grouped by label with a per-label cap."""
    if not schema_id:
        raise DatabaseError("schema_id cannot be empty")
    if limit_per_label <= 0:
        return []

    query = """
        select
            feedback_id,
            schema_id,
            resume_id,
            label,
            feedback_text,
            score,
            scoring_result,
            created_at
        from (
            select
                feedback_id,
                schema_id,
                resume_id,
                label,
                feedback_text,
                score,
                scoring_result,
                created_at,
                row_number() over (
                    partition by label
                    order by created_at desc, feedback_id
                ) as rn
            from feedback_examples
            where schema_id = %(schema_id)s
        ) ranked
        where rn <= %(limit_per_label)s
        order by label, created_at desc, feedback_id
    """

    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "schema_id": schema_id,
                        "limit_per_label": limit_per_label,
                    },
                )
                rows = cur.fetchall()
            return [
                {
                    "feedback_id": row[0],
                    "schema_id": row[1],
                    "resume_id": row[2],
                    "label": row[3],
                    "feedback_text": row[4],
                    "score": float(row[5]) if row[5] is not None else None,
                    "scoring_result": row[6] if isinstance(row[6], dict) else None,
                    "created_at": row[7].isoformat() if row[7] is not None else None,
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to fetch feedback examples: {exc}") from exc


def get_resumes_by_ids(resume_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch resume records used by scoring prompts."""
    cleaned_ids = [resume_id for resume_id in resume_ids if resume_id]
    if not cleaned_ids:
        return []

    query = """
        select resume_id, name, metadata, semantic_text, raw_json
        from resumes
        where resume_id = any(%(resume_ids)s::text[])
        order by array_position(%(resume_ids)s::text[], resume_id)
    """

    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(query, {"resume_ids": cleaned_ids})
                rows = cur.fetchall()
            return [
                {
                    "resume_id": row[0],
                    "candidate_name": (row[1] or "").strip(),
                    "metadata": row[2] if isinstance(row[2], dict) else {},
                    "semantic_text": row[3] or "",
                    "raw_json": row[4] if isinstance(row[4], dict) else {},
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to fetch resumes: {exc}") from exc


def save_scoring_feedback(
    *,
    feedback_id: str,
    schema_id: str,
    resume_id: str,
    label: str,
    feedback_text: Optional[str] = None,
    score: Optional[float] = None,
    scoring_result: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Persist human feedback for a schema/resume scoring result.

    Feedback is one row per schema/resume pair. If a user updates feedback for
    the same resume under the same schema, keep the newest judgment instead of
    accumulating duplicates.
    """
    if label not in {"excellent", "good", "qualified", "bad", "n/a"}:
        raise DatabaseError("label must be one of: excellent, good, qualified, bad, n/a")

    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                params = {
                    "feedback_id": feedback_id,
                    "schema_id": schema_id,
                    "resume_id": resume_id,
                    "label": label,
                    "feedback_text": feedback_text,
                    "score": score,
                    "scoring_result": json.dumps(scoring_result, ensure_ascii=False)
                    if scoring_result is not None
                    else None,
                }
                cur.execute(
                    """
                    select feedback_id
                    from feedback_examples
                    where schema_id = %(schema_id)s
                      and resume_id = %(resume_id)s
                    order by updated_at desc, created_at desc
                    limit 1
                    """,
                    params,
                )
                existing = cur.fetchone()

                if existing is not None:
                    feedback_id = existing[0]
                    params["feedback_id"] = feedback_id
                    cur.execute(
                        """
                        delete from feedback_examples
                        where schema_id = %(schema_id)s
                          and resume_id = %(resume_id)s
                          and feedback_id <> %(feedback_id)s
                        """,
                        params,
                    )
                    cur.execute(
                        """
                        update feedback_examples
                        set
                            label = %(label)s,
                            feedback_text = %(feedback_text)s,
                            score = %(score)s,
                            scoring_result = %(scoring_result)s::jsonb,
                            updated_at = now()
                        where feedback_id = %(feedback_id)s
                        returning created_at, updated_at
                        """,
                        params,
                    )
                    created_at, updated_at = cur.fetchone()
                    action = "updated"
                else:
                    cur.execute(
                        """
                        insert into feedback_examples (
                            feedback_id,
                            schema_id,
                            resume_id,
                            label,
                            feedback_text,
                            score,
                            scoring_result
                        )
                        values (
                            %(feedback_id)s,
                            %(schema_id)s,
                            %(resume_id)s,
                            %(label)s,
                            %(feedback_text)s,
                            %(score)s,
                            %(scoring_result)s::jsonb
                        )
                        returning created_at, updated_at
                        """,
                        params,
                    )
                    created_at, updated_at = cur.fetchone()
                    action = "created"
            conn.commit()
            return {
                "feedback_id": feedback_id,
                "schema_id": schema_id,
                "resume_id": resume_id,
                "label": label,
                "feedback_text": feedback_text,
                "score": score,
                "action": action,
                "created_at": created_at.isoformat() if created_at is not None else None,
                "updated_at": updated_at.isoformat() if updated_at is not None else None,
            }
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to save scoring feedback: {exc}") from exc


# ─────────────────────────────────────────────────────────────
# Chat session CRUD
# ─────────────────────────────────────────────────────────────

def create_chat_session(
    *,
    session_id: str,
    title: Optional[str] = None,
    active_schema_id: Optional[str] = None,
    hr_auto_mode: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into chat_sessions (session_id, title, active_schema_id, hr_auto_mode, provider, model)
                    values (%(session_id)s, %(title)s, %(active_schema_id)s, %(hr_auto_mode)s, %(provider)s, %(model)s)
                    returning created_at, updated_at
                    """,
                    {
                        "session_id": session_id,
                        "title": title,
                        "active_schema_id": active_schema_id,
                        "hr_auto_mode": hr_auto_mode,
                        "provider": provider,
                        "model": model,
                    },
                )
                created_at, updated_at = cur.fetchone()
            conn.commit()
            return {
                "session_id": session_id,
                "title": title,
                "active_schema_id": active_schema_id,
                "hr_auto_mode": hr_auto_mode,
                "provider": provider,
                "model": model,
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
            }
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to create chat session: {exc}") from exc


def load_chat_session(session_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select session_id, title, active_schema_id, hr_auto_mode, provider, model,
                           created_at, updated_at
                    from chat_sessions where session_id = %(session_id)s
                    """,
                    {"session_id": session_id},
                )
                row = cur.fetchone()
            if row is None:
                return None
            return {
                "session_id": row[0],
                "title": row[1],
                "active_schema_id": row[2],
                "hr_auto_mode": row[3],
                "provider": row[4],
                "model": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "updated_at": row[7].isoformat() if row[7] else None,
            }
        except Exception as exc:
            raise DatabaseError(f"Failed to load chat session: {exc}") from exc


def save_chat_session(
    *,
    session_id: str,
    title: Optional[str] = None,
    active_schema_id: Optional[str] = None,
    hr_auto_mode: Optional[bool] = None,
) -> None:
    fields: list[str] = ["updated_at = now()"]
    params: dict[str, Any] = {"session_id": session_id}
    if title is not None:
        fields.append("title = %(title)s")
        params["title"] = title
    if active_schema_id is not None:
        fields.append("active_schema_id = %(active_schema_id)s")
        params["active_schema_id"] = active_schema_id
    if hr_auto_mode is not None:
        fields.append("hr_auto_mode = %(hr_auto_mode)s")
        params["hr_auto_mode"] = hr_auto_mode

    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"update chat_sessions set {', '.join(fields)} where session_id = %(session_id)s",
                    params,
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to update chat session: {exc}") from exc


def append_chat_message(
    *,
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_calls_json: Optional[list] = None,
) -> str:
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into chat_messages
                        (message_id, session_id, role, content, tool_name, tool_call_id, tool_calls_json)
                    values
                        (%(message_id)s, %(session_id)s, %(role)s, %(content)s,
                         %(tool_name)s, %(tool_call_id)s, %(tool_calls_json)s::jsonb)
                    """,
                    {
                        "message_id": message_id,
                        "session_id": session_id,
                        "role": role,
                        "content": content,
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "tool_calls_json": json.dumps(tool_calls_json) if tool_calls_json else None,
                    },
                )
            conn.commit()
            return message_id
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to append chat message: {exc}") from exc


def get_chat_messages(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select message_id, session_id, role, content, tool_name, tool_call_id,
                           tool_calls_json, created_at
                    from chat_messages
                    where session_id = %(session_id)s
                    order by created_at asc, message_id asc
                    limit %(limit)s
                    """,
                    {"session_id": session_id, "limit": limit},
                )
                rows = cur.fetchall()
            return [
                {
                    "message_id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "tool_name": row[4],
                    "tool_call_id": row[5],
                    "tool_calls_json": row[6] if isinstance(row[6], list) else None,
                    "created_at": row[7].isoformat() if row[7] else None,
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to fetch chat messages: {exc}") from exc


def list_chat_sessions(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select session_id, title, active_schema_id, hr_auto_mode, provider, model,
                           created_at, updated_at
                    from chat_sessions
                    order by updated_at desc, created_at desc
                    limit %(limit)s
                    """,
                    {"limit": limit},
                )
                rows = cur.fetchall()
            return [
                {
                    "session_id": row[0],
                    "title": row[1],
                    "active_schema_id": row[2],
                    "hr_auto_mode": row[3],
                    "provider": row[4],
                    "model": row[5],
                    "created_at": row[6].isoformat() if row[6] else None,
                    "updated_at": row[7].isoformat() if row[7] else None,
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to list chat sessions: {exc}") from exc


def delete_chat_session(session_id: str) -> None:
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "delete from chat_sessions where session_id = %(session_id)s",
                    {"session_id": session_id},
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to delete chat session: {exc}") from exc


def get_scoring_schema_by_id(schema_id: str) -> Optional[dict[str, Any]]:
    """Fetch a single scoring schema by its ID."""
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select schema_id, schema_name, rules_json, summary, version, is_active
                    from scoring_schemas
                    where schema_id = %(schema_id)s
                    """,
                    {"schema_id": schema_id},
                )
                row = cur.fetchone()
            if row is None:
                return None
            return {
                "schema_id": row[0],
                "schema_name": row[1],
                "rules_json": row[2] if isinstance(row[2], dict) else {},
                "summary": row[3],
                "version": row[4],
                "is_active": row[5],
            }
        except Exception as exc:
            raise DatabaseError(f"Failed to fetch scoring schema: {exc}") from exc


# ─────────────────────────────────────────────────────────────
# Agent memory CRUD
# ─────────────────────────────────────────────────────────────

def store_memory(
    *,
    memory_id: str,
    session_id: Optional[str],
    memory_type: str,
    content: str,
    content_json: Optional[dict] = None,
    embedding: Optional[list[float]] = None,
    expires_at: Optional[str] = None,  # ISO timestamp or None
) -> dict[str, Any]:
    """Insert or replace a single memory entry."""
    embedding_dimension = _embedding_dimension(embedding)
    with _connect() as conn:
        _ensure_schema(conn, embedding_dimension)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into agent_memories
                        (memory_id, session_id, memory_type, content, content_json,
                         embedding, expires_at)
                    values
                        (%(memory_id)s, %(session_id)s, %(memory_type)s, %(content)s,
                         %(content_json)s::jsonb, %(embedding)s::vector, %(expires_at)s::timestamptz)
                    on conflict (memory_id) do update
                    set content      = excluded.content,
                        content_json = excluded.content_json,
                        embedding    = excluded.embedding,
                        expires_at   = excluded.expires_at,
                        updated_at   = now()
                    returning created_at, updated_at
                    """,
                    {
                        "memory_id": memory_id,
                        "session_id": session_id,
                        "memory_type": memory_type,
                        "content": content,
                        "content_json": json.dumps(content_json, ensure_ascii=False) if content_json else None,
                        "embedding": _vector_literal(embedding),
                        "expires_at": expires_at,
                    },
                )
                created_at, updated_at = cur.fetchone()
            conn.commit()
            return {
                "memory_id": memory_id,
                "session_id": session_id,
                "memory_type": memory_type,
                "content": content,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            }
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to store memory: {exc}") from exc


def retrieve_memories_by_embedding(
    query_embedding: list[float],
    *,
    memory_types: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    limit: int = 5,
    exclude_expired: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve memories ordered by cosine similarity to the query embedding."""
    if not query_embedding:
        return []

    where_parts = ["embedding is not null"]
    params: dict[str, Any] = {
        "query_embedding": _vector_literal(query_embedding),
        "limit": limit,
    }
    if exclude_expired:
        where_parts.append("(expires_at is null or expires_at > now())")
    if memory_types:
        where_parts.append("memory_type = any(%(memory_types)s::text[])")
        params["memory_types"] = memory_types
    if session_id:
        where_parts.append("(session_id = %(session_id)s or session_id is null)")
        params["session_id"] = session_id

    where_clause = " and ".join(where_parts)
    query = f"""
        select memory_id, session_id, memory_type, content, content_json, expires_at,
               created_at, 1 - (embedding <=> %(query_embedding)s::vector) as similarity
        from agent_memories
        where {where_clause}
        order by embedding <=> %(query_embedding)s::vector
        limit %(limit)s
    """

    with _connect() as conn:
        _ensure_schema(conn, len(query_embedding))
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return [
                {
                    "memory_id": row[0],
                    "session_id": row[1],
                    "memory_type": row[2],
                    "content": row[3],
                    "content_json": row[4] if isinstance(row[4], dict) else None,
                    "expires_at": row[5].isoformat() if row[5] else None,
                    "created_at": row[6].isoformat() if row[6] else None,
                    "similarity": float(row[7]) if row[7] is not None else 0.0,
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to retrieve memories: {exc}") from exc


def get_memories_by_session(
    session_id: str,
    *,
    memory_types: Optional[list[str]] = None,
    exclude_expired: bool = True,
) -> list[dict[str, Any]]:
    """Fetch all memories for a given session."""
    where_parts = ["session_id = %(session_id)s"]
    params: dict[str, Any] = {"session_id": session_id}
    if exclude_expired:
        where_parts.append("(expires_at is null or expires_at > now())")
    if memory_types:
        where_parts.append("memory_type = any(%(memory_types)s::text[])")
        params["memory_types"] = memory_types

    where_clause = " and ".join(where_parts)
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    select memory_id, session_id, memory_type, content, content_json,
                           expires_at, created_at
                    from agent_memories
                    where {where_clause}
                    order by created_at desc
                    """,
                    params,
                )
                rows = cur.fetchall()
            return [
                {
                    "memory_id": row[0],
                    "session_id": row[1],
                    "memory_type": row[2],
                    "content": row[3],
                    "content_json": row[4] if isinstance(row[4], dict) else None,
                    "expires_at": row[5].isoformat() if row[5] else None,
                    "created_at": row[6].isoformat() if row[6] else None,
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to get session memories: {exc}") from exc


def invalidate_session_search_cache(session_id: str) -> None:
    """Expire all search_result_cache memories for a session (called when schema changes)."""
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update agent_memories
                    set expires_at = now(), updated_at = now()
                    where session_id = %(session_id)s
                      and memory_type = 'search_result_cache'
                      and (expires_at is null or expires_at > now())
                    """,
                    {"session_id": session_id},
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to invalidate search cache: {exc}") from exc


def get_cached_search_results(
    session_id: str, query_hash: str
) -> Optional[dict[str, Any]]:
    """Return cached search results or None if expired/missing."""
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select content_json
                    from agent_memories
                    where session_id = %(session_id)s
                      and memory_type = 'search_result_cache'
                      and content = %(query_hash)s
                      and (expires_at is null or expires_at > now())
                    order by created_at desc
                    limit 1
                    """,
                    {"session_id": session_id, "query_hash": query_hash},
                )
                row = cur.fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row[0], dict) else None
        except Exception as exc:
            raise DatabaseError(f"Failed to get cached search results: {exc}") from exc


def cache_search_results(
    session_id: str,
    query_hash: str,
    results: dict[str, Any],
    ttl_hours: int = 1,
) -> None:
    """Store search results in the memory cache with a TTL."""
    from datetime import datetime, timezone, timedelta
    memory_id = f"sc_{session_id}_{query_hash[:16]}"
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
    store_memory(
        memory_id=memory_id,
        session_id=session_id,
        memory_type="search_result_cache",
        content=query_hash,
        content_json=results,
        expires_at=expires_at,
    )


def find_similar_session(
    query_embedding: list[float],
    *,
    similarity_threshold: float = 0.85,
    window_days: int = 7,
) -> Optional[dict[str, Any]]:
    """
    Find a recent session whose summary is semantically close to the query.
    Returns the session_id + similarity or None.
    """
    if not query_embedding:
        return None
    with _connect() as conn:
        _ensure_schema(conn, len(query_embedding))
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select m.session_id,
                           1 - (m.embedding <=> %(query_embedding)s::vector) as similarity
                    from agent_memories m
                    join chat_sessions s on s.session_id = m.session_id
                    where m.memory_type = 'session_summary'
                      and m.embedding is not null
                      and m.created_at > now() - %(window)s::interval
                      and (m.expires_at is null or m.expires_at > now())
                    order by m.embedding <=> %(query_embedding)s::vector
                    limit 1
                    """,
                    {
                        "query_embedding": _vector_literal(query_embedding),
                        "window": f"{window_days} days",
                    },
                )
                row = cur.fetchone()
            if row is None:
                return None
            similarity = float(row[1]) if row[1] is not None else 0.0
            if similarity >= similarity_threshold:
                return {"session_id": row[0], "similarity": similarity}
            return None
        except Exception as exc:
            raise DatabaseError(f"Failed to find similar session: {exc}") from exc


def list_scoring_schemas(limit: int = 20) -> list[dict[str, Any]]:
    """List active scoring schemas ordered by most recently updated."""
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select schema_id, schema_name, summary, version, is_active, created_at, updated_at
                    from scoring_schemas
                    where is_active = true
                    order by updated_at desc, created_at desc
                    limit %(limit)s
                    """,
                    {"limit": limit},
                )
                rows = cur.fetchall()
            return [
                {
                    "schema_id": row[0],
                    "schema_name": row[1],
                    "summary": row[2],
                    "version": row[3],
                    "is_active": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                    "updated_at": row[6].isoformat() if row[6] else None,
                }
                for row in rows
            ]
        except Exception as exc:
            raise DatabaseError(f"Failed to list scoring schemas: {exc}") from exc


def get_resume_storage_info(resume_id: str) -> Optional[dict[str, Any]]:
    """Return S3 storage fields for a resume, or None if not found / not stored."""
    with _connect() as conn:
        _ensure_schema(conn, settings.EMBEDDING_DIMENSION)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select pdf_storage_bucket, pdf_storage_key, pdf_mime_type
                    from resumes
                    where resume_id = %(resume_id)s
                    """,
                    {"resume_id": resume_id},
                )
                row = cur.fetchone()
            if not row or not row[0] or not row[1]:
                return None
            return {
                "pdf_storage_bucket": row[0],
                "pdf_storage_key": row[1],
                "pdf_mime_type": row[2],
            }
        except Exception as exc:
            raise DatabaseError(f"Failed to get resume storage info: {exc}") from exc
