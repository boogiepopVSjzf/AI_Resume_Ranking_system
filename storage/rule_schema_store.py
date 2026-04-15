from __future__ import annotations

import json
from typing import Optional

from config import settings
from schemas.final_result import RuleSchemaResponse
from utils.errors import DatabaseError
from utils.logger import get_logger

logger = get_logger("rule_schema_store")

_RULE_SCHEMA_READY = False


def _vector_literal(vector: Optional[list[float]]) -> Optional[str]:
    if vector is None:
        return None
    if not vector:
        raise DatabaseError("Embedding vector is empty")
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def _embedding_dimension(vector: list[float]) -> int:
    if not vector:
        raise DatabaseError("Embedding vector is empty")
    return len(vector)


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


def _ensure_rule_schema_dataset(conn, embedding_dimension: int) -> None:
    global _RULE_SCHEMA_READY
    if _RULE_SCHEMA_READY or not settings.DB_AUTO_INIT:
        return

    if embedding_dimension <= 0:
        raise DatabaseError("Embedding dimension must be a positive integer")

    statements = [
        "create extension if not exists vector",
        f"""
        create table if not exists rule_schemas (
            schema_id text primary key,
            rule_json jsonb not null,
            rule_description text not null,
            embedding vector({embedding_dimension}) not null,
            job_name text not null,
            created_at timestamptz not null default now(),
            updated_at timestamptz not null default now()
        )
        """,
        "alter table rule_schemas add column if not exists rule_json jsonb not null default '{}'::jsonb",
        "alter table rule_schemas add column if not exists rule_description text not null default ''",
        "alter table rule_schemas add column if not exists job_name text not null default ''",
        "alter table rule_schemas add column if not exists created_at timestamptz not null default now()",
        "alter table rule_schemas add column if not exists updated_at timestamptz not null default now()",
        "create index if not exists idx_rule_schemas_job_name on rule_schemas (job_name)",
        "create index if not exists idx_rule_schemas_rule_json_gin on rule_schemas using gin (rule_json)",
        "create index if not exists idx_rule_schemas_embedding_hnsw on rule_schemas using hnsw (embedding vector_cosine_ops)",
    ]

    try:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
        conn.commit()
        _RULE_SCHEMA_READY = True
    except Exception as exc:
        conn.rollback()
        raise DatabaseError(f"Failed to initialise rule_schemas dataset: {exc}") from exc


def save_rule_schema_result(result: RuleSchemaResponse) -> None:
    """Upsert one generated rule schema row into Postgres."""
    embedding_dimension = _embedding_dimension(result.embedding_vector)

    with _connect() as conn:
        _ensure_rule_schema_dataset(conn, embedding_dimension)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into rule_schemas (
                        schema_id,
                        rule_json,
                        rule_description,
                        embedding,
                        job_name
                    )
                    values (
                        %(schema_id)s,
                        %(rule_json)s::jsonb,
                        %(rule_description)s,
                        %(embedding)s::vector,
                        %(job_name)s
                    )
                    on conflict (schema_id) do update
                    set
                        rule_json = excluded.rule_json,
                        rule_description = excluded.rule_description,
                        embedding = excluded.embedding,
                        job_name = excluded.job_name,
                        updated_at = now()
                    """,
                    {
                        "schema_id": result.schema_id,
                        "rule_json": json.dumps(result.rule_json, ensure_ascii=False),
                        "rule_description": result.rule_description,
                        "embedding": _vector_literal(result.embedding_vector),
                        "job_name": result.job_name,
                    },
                )
            conn.commit()
            logger.info("Persisted rule schema %s to Postgres", result.schema_id)
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"Failed to save rule schema result: {exc}") from exc
