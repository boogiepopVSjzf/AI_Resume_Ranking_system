"""
Agent tool definitions and executors for the multi-turn HR conversation agent.

Each tool has:
  TOOL_DEF  — the JSON-Schema descriptor sent to the LLM
  _exec_*   — async executor that calls existing services and returns a JSON string
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from config import settings
from services.embedding_service import embed_text
from services.job_query_rewrite_service import rewrite_merged_context
from services.scoring_schema_service import build_scoring_schema_payload, parse_rules_text
from services.scoring_service import (
    build_feedback_calibration_data,
    normalize_feedback_influence_mode,
    score_resume_with_schema,
)
from storage.postgres_store import (
    get_feedback_examples,
    get_resumes_by_ids,
    get_scoring_schema_by_id,
    list_scoring_schemas,
    query_resume_ids_by_hard_filters,
    query_similar_resumes,
    save_scoring_schema,
)
from utils.errors import LLMError, LLMParseError, DatabaseError
from utils.logger import get_logger

if TYPE_CHECKING:
    from services.session_service import SessionContext

logger = get_logger("agent_tools")


@dataclass
class ToolResult:
    tool_call_id: str
    tool_name: str
    output: str             # JSON string returned to LLM
    is_error: bool = False
    requires_confirmation: bool = False
    confirmation_payload: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# Tool definitions (provider-agnostic, input_schema field)
# ─────────────────────────────────────────────────────────────

TOOL_SEARCH_CANDIDATES: dict = {
    "name": "search_candidates",
    "description": (
        "Search the resume database for candidates matching a job description. "
        "Returns a ranked list of candidate IDs with similarity scores. "
        "Use this before scoring to get candidate IDs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_description": {
                "type": "string",
                "description": "The job description or HR note describing the ideal candidate.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of candidates to return (default 10).",
                "default": 10,
            },
            "filter_mode": {
                "type": "string",
                "enum": ["strict", "balanced", "semantic_only"],
                "description": "How aggressively to apply hard filters. Default: balanced.",
                "default": "balanced",
            },
        },
        "required": ["job_description"],
    },
}

TOOL_SCORE_CANDIDATES: dict = {
    "name": "score_candidates",
    "description": (
        "Score a list of candidates using the currently active scoring schema. "
        "Returns scores and rule-by-rule explanations. "
        "Requires an active schema — call get_active_schema first or create one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resume_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of resume IDs to score.",
            },
            "feedback_influence_mode": {
                "type": "string",
                "enum": ["on", "off"],
                "description": "Whether to use past feedback to calibrate scores. Default: on.",
                "default": "on",
            },
        },
        "required": ["resume_ids"],
    },
}

TOOL_DRAFT_SCHEMA: dict = {
    "name": "draft_schema",
    "description": (
        "Create a draft scoring schema from a textual description of requirements. "
        "The draft is shown to the HR user for confirmation before being saved. "
        "Rules must be provided in the format: "
        "'rules1: <requirement text> weight: 40%, rules2: <text> weight: 60%'. "
        "Weights must sum to 100%."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "schema_name": {
                "type": "string",
                "description": "A short descriptive name for this scoring schema.",
            },
            "rules_text": {
                "type": "string",
                "description": (
                    "Scoring rules in the format: "
                    "'rules1: <description> weight: X%, rules2: <description> weight: Y%'"
                ),
            },
        },
        "required": ["schema_name", "rules_text"],
    },
}

TOOL_SAVE_SCHEMA: dict = {
    "name": "save_schema",
    "description": (
        "Save the pending schema draft to the database after HR confirmation. "
        "Only call this after draft_schema returned a confirmation_required event "
        "and the HR user has confirmed. "
        "If HR auto_mode is enabled, this is called automatically after draft_schema."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "schema_name": {"type": "string"},
            "rules_text": {"type": "string"},
        },
        "required": ["schema_name", "rules_text"],
    },
}

TOOL_EXPLAIN_SCORE: dict = {
    "name": "explain_score",
    "description": (
        "Provide a detailed explanation of why a specific candidate received their score. "
        "Returns per-rule reasoning."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "resume_id": {
                "type": "string",
                "description": "The resume ID to explain.",
            },
        },
        "required": ["resume_id"],
    },
}

TOOL_GET_ACTIVE_SCHEMA: dict = {
    "name": "get_active_schema",
    "description": "Get the currently active scoring schema for this session.",
    "input_schema": {"type": "object", "properties": {}},
}

TOOL_SET_ACTIVE_SCHEMA: dict = {
    "name": "set_active_schema",
    "description": "Set the active scoring schema for this session by schema ID.",
    "input_schema": {
        "type": "object",
        "properties": {
            "schema_id": {
                "type": "string",
                "description": "The schema_id to activate.",
            },
        },
        "required": ["schema_id"],
    },
}

TOOL_LIST_SCHEMAS: dict = {
    "name": "list_schemas",
    "description": "List all available active scoring schemas in the database.",
    "input_schema": {"type": "object", "properties": {}},
}

TOOL_GET_CANDIDATE: dict = {
    "name": "get_candidate",
    "description": "Retrieve details of one or more candidates by resume ID.",
    "input_schema": {
        "type": "object",
        "properties": {
            "resume_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of resume IDs to fetch.",
            },
        },
        "required": ["resume_ids"],
    },
}

ALL_TOOLS: list[dict] = [
    TOOL_SEARCH_CANDIDATES,
    TOOL_SCORE_CANDIDATES,
    TOOL_DRAFT_SCHEMA,
    TOOL_SAVE_SCHEMA,
    TOOL_EXPLAIN_SCORE,
    TOOL_GET_ACTIVE_SCHEMA,
    TOOL_SET_ACTIVE_SCHEMA,
    TOOL_LIST_SCHEMAS,
    TOOL_GET_CANDIDATE,
]


# ─────────────────────────────────────────────────────────────
# Tool executors
# ─────────────────────────────────────────────────────────────

def _run_sync(fn, *args, **kwargs):
    """Run a synchronous function in an executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def _exec_search_candidates(args: dict, ctx: "SessionContext") -> ToolResult:
    job_description = args.get("job_description", "")
    top_k = int(args.get("top_k", 10))
    filter_mode = args.get("filter_mode", "balanced")

    # ── Check session search cache ────────────────────────────────────────
    from services.memory_agent import get_search_cache, make_search_cache_key, set_search_cache
    cache_key = make_search_cache_key(job_description, filter_mode, top_k)
    cached = get_search_cache(ctx.session_id, cache_key) if settings.ENABLE_DB_PERSISTENCE else None
    if cached is not None:
        logger.info("search_candidates cache hit for session %s", ctx.session_id)
        ctx.recent_search_result_ids = [r["resume_id"] for r in cached.get("results", [])]
        cached["_cached"] = True
        return ToolResult(
            tool_call_id="",
            tool_name="search_candidates",
            output=json.dumps(cached, ensure_ascii=False),
        )

    try:
        query, _ = await _run_sync(
            rewrite_merged_context, job_description, filter_mode=filter_mode
        )

        if filter_mode == "semantic_only":
            from schemas.job_query import HardFilters
            hard_filters = HardFilters()
        else:
            hard_filters = query.hard_filters

        resume_ids = await _run_sync(query_resume_ids_by_hard_filters, hard_filters)
        search_embedding = await _run_sync(embed_text, query.search_query)
        results = await _run_sync(
            query_similar_resumes,
            resume_ids=resume_ids,
            search_query_embedding=search_embedding,
            top_k=top_k,
        )

        ctx.recent_search_result_ids = [r.resume_id for r in results]

        output = {
            "search_query": query.search_query,
            "hard_filters_applied": hard_filters.model_dump(),
            "total_filtered": len(resume_ids),
            "results": [
                {
                    "resume_id": r.resume_id,
                    "similarity_score": round(r.similarity_score, 4),
                    "name": r.metadata.get("name") or r.raw_json.get("name") or "Unknown",
                }
                for r in results
            ],
        }

        # Cache for this session
        if settings.ENABLE_DB_PERSISTENCE:
            set_search_cache(ctx.session_id, cache_key, output)

        return ToolResult(
            tool_call_id="",
            tool_name="search_candidates",
            output=json.dumps(output, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("search_candidates failed: %s", exc)
        return ToolResult(
            tool_call_id="",
            tool_name="search_candidates",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_score_candidates(args: dict, ctx: "SessionContext") -> ToolResult:
    resume_ids: list[str] = args.get("resume_ids", [])
    feedback_influence_mode = normalize_feedback_influence_mode(
        args.get("feedback_influence_mode", "on")
    )

    if not ctx.active_schema_id:
        return ToolResult(
            tool_call_id="",
            tool_name="score_candidates",
            output=json.dumps({"error": "No active schema. Please create or select a schema first."}),
            is_error=True,
        )

    try:
        schema = await _run_sync(get_scoring_schema_by_id, ctx.active_schema_id)
        if schema is None:
            return ToolResult(
                tool_call_id="",
                tool_name="score_candidates",
                output=json.dumps({"error": f"Schema {ctx.active_schema_id} not found."}),
                is_error=True,
            )

        feedback_examples = await _run_sync(
            get_feedback_examples, schema_id=ctx.active_schema_id, limit_per_label=2
        )
        resumes = await _run_sync(get_resumes_by_ids, resume_ids)

        results = []
        for resume in resumes:
            score, _ = await _run_sync(
                score_resume_with_schema,
                schema=schema,
                feedback_examples=feedback_examples,
                resume=resume,
                feedback_influence_mode=feedback_influence_mode,
            )
            results.append({
                "resume_id": resume["resume_id"],
                "candidate_name": resume.get("candidate_name", "Unknown"),
                "score": round(score.score, 2),
                "explanation": score.explanation,
                "rule_scores": {
                    k: {"score": round(v.score, 2), "reason": v.reason, "weight": v.weight}
                    for k, v in score.rule_scores.items()
                },
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        ctx.recent_score_results = results

        return ToolResult(
            tool_call_id="",
            tool_name="score_candidates",
            output=json.dumps({
                "schema_name": schema["schema_name"],
                "schema_id": ctx.active_schema_id,
                "results": results,
            }, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("score_candidates failed: %s", exc)
        return ToolResult(
            tool_call_id="",
            tool_name="score_candidates",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_draft_schema(args: dict, ctx: "SessionContext") -> ToolResult:
    schema_name = (args.get("schema_name") or "").strip()
    rules_text = (args.get("rules_text") or "").strip()

    if not schema_name or not rules_text:
        return ToolResult(
            tool_call_id="",
            tool_name="draft_schema",
            output=json.dumps({"error": "schema_name and rules_text are required."}),
            is_error=True,
        )

    try:
        payload, _ = await _run_sync(build_scoring_schema_payload, schema_name, rules_text)
        draft = {
            "schema_name": payload["schema_name"],
            "rules_text": rules_text,
            "rules_json": payload["rules_json"],
            "summary": payload["summary"],
        }
        ctx.pending_confirmation = draft

        if ctx.hr_auto_mode:
            return await _exec_save_schema(
                {"schema_name": schema_name, "rules_text": rules_text}, ctx
            )

        return ToolResult(
            tool_call_id="",
            tool_name="draft_schema",
            output=json.dumps({"status": "draft_ready", "draft": draft}, ensure_ascii=False),
            requires_confirmation=True,
            confirmation_payload=draft,
        )
    except (ValueError, LLMParseError) as exc:
        return ToolResult(
            tool_call_id="",
            tool_name="draft_schema",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )
    except Exception as exc:
        logger.warning("draft_schema failed: %s", exc)
        return ToolResult(
            tool_call_id="",
            tool_name="draft_schema",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_save_schema(args: dict, ctx: "SessionContext") -> ToolResult:
    schema_name = (args.get("schema_name") or "").strip()
    rules_text = (args.get("rules_text") or "").strip()

    # If a draft was pending, use it directly to avoid regenerating the summary
    pending = ctx.pending_confirmation
    if pending and pending.get("schema_name") == schema_name:
        payload_data = pending
        embedding = await _run_sync(embed_text, pending["summary"])
    else:
        try:
            p, _ = await _run_sync(build_scoring_schema_payload, schema_name, rules_text)
            payload_data = p
            embedding = p["embedding"]
        except Exception as exc:
            return ToolResult(
                tool_call_id="",
                tool_name="save_schema",
                output=json.dumps({"error": str(exc)}),
                is_error=True,
            )

    schema_id = uuid4().hex
    try:
        persisted = await _run_sync(
            save_scoring_schema,
            schema_id=schema_id,
            schema_name=payload_data["schema_name"],
            rules_json=payload_data["rules_json"],
            summary=payload_data["summary"],
            embedding=embedding,
        )
        ctx.active_schema_id = persisted["schema_id"]
        ctx.pending_confirmation = None

        from services.session_service import update_active_schema
        update_active_schema(ctx, persisted["schema_id"])

        # Invalidate stale search cache — new schema means previous scores are irrelevant
        if settings.ENABLE_DB_PERSISTENCE:
            from services.memory_agent import invalidate_search_cache
            invalidate_search_cache(ctx.session_id)

        return ToolResult(
            tool_call_id="",
            tool_name="save_schema",
            output=json.dumps({
                "status": "saved",
                "schema_id": persisted["schema_id"],
                "schema_name": persisted["schema_name"],
                "version": persisted["version"],
                "summary": persisted.get("summary", ""),
            }, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("save_schema failed: %s", exc)
        return ToolResult(
            tool_call_id="",
            tool_name="save_schema",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_explain_score(args: dict, ctx: "SessionContext") -> ToolResult:
    resume_id = args.get("resume_id", "")
    if not resume_id:
        return ToolResult(
            tool_call_id="",
            tool_name="explain_score",
            output=json.dumps({"error": "resume_id is required."}),
            is_error=True,
        )
    if not ctx.active_schema_id:
        return ToolResult(
            tool_call_id="",
            tool_name="explain_score",
            output=json.dumps({"error": "No active schema set."}),
            is_error=True,
        )

    try:
        schema = await _run_sync(get_scoring_schema_by_id, ctx.active_schema_id)
        if schema is None:
            return ToolResult(
                tool_call_id="",
                tool_name="explain_score",
                output=json.dumps({"error": "Schema not found."}),
                is_error=True,
            )
        resumes = await _run_sync(get_resumes_by_ids, [resume_id])
        if not resumes:
            return ToolResult(
                tool_call_id="",
                tool_name="explain_score",
                output=json.dumps({"error": f"Resume {resume_id} not found."}),
                is_error=True,
            )
        feedback_examples = await _run_sync(
            get_feedback_examples, schema_id=ctx.active_schema_id, limit_per_label=2
        )
        score, _ = await _run_sync(
            score_resume_with_schema,
            schema=schema,
            feedback_examples=feedback_examples,
            resume=resumes[0],
            feedback_influence_mode="on",
        )
        return ToolResult(
            tool_call_id="",
            tool_name="explain_score",
            output=json.dumps({
                "resume_id": resume_id,
                "candidate_name": resumes[0].get("candidate_name", "Unknown"),
                "score": round(score.score, 2),
                "explanation": score.explanation,
                "rule_scores": {
                    k: {
                        "score": round(v.score, 2),
                        "reason": v.reason,
                        "weight": v.weight,
                        "weighted_score": round(v.weighted_score, 2),
                    }
                    for k, v in score.rule_scores.items()
                },
            }, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("explain_score failed: %s", exc)
        return ToolResult(
            tool_call_id="",
            tool_name="explain_score",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_get_active_schema(args: dict, ctx: "SessionContext") -> ToolResult:
    if not ctx.active_schema_id:
        return ToolResult(
            tool_call_id="",
            tool_name="get_active_schema",
            output=json.dumps({"active_schema": None, "message": "No active schema in this session."}),
        )
    try:
        schema = await _run_sync(get_scoring_schema_by_id, ctx.active_schema_id)
        return ToolResult(
            tool_call_id="",
            tool_name="get_active_schema",
            output=json.dumps({"active_schema": schema}, ensure_ascii=False),
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            tool_name="get_active_schema",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_set_active_schema(args: dict, ctx: "SessionContext") -> ToolResult:
    schema_id = args.get("schema_id", "")
    if not schema_id:
        return ToolResult(
            tool_call_id="",
            tool_name="set_active_schema",
            output=json.dumps({"error": "schema_id is required."}),
            is_error=True,
        )
    try:
        schema = await _run_sync(get_scoring_schema_by_id, schema_id)
        if schema is None:
            return ToolResult(
                tool_call_id="",
                tool_name="set_active_schema",
                output=json.dumps({"error": f"Schema {schema_id} not found."}),
                is_error=True,
            )
        from services.session_service import update_active_schema
        update_active_schema(ctx, schema_id)
        return ToolResult(
            tool_call_id="",
            tool_name="set_active_schema",
            output=json.dumps({
                "status": "ok",
                "active_schema_id": schema_id,
                "schema_name": schema["schema_name"],
            }),
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            tool_name="set_active_schema",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_list_schemas(args: dict, ctx: "SessionContext") -> ToolResult:
    try:
        schemas = await _run_sync(list_scoring_schemas, 20)
        return ToolResult(
            tool_call_id="",
            tool_name="list_schemas",
            output=json.dumps({"schemas": schemas}, ensure_ascii=False),
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            tool_name="list_schemas",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


async def _exec_get_candidate(args: dict, ctx: "SessionContext") -> ToolResult:
    resume_ids: list[str] = args.get("resume_ids", [])
    if not resume_ids:
        return ToolResult(
            tool_call_id="",
            tool_name="get_candidate",
            output=json.dumps({"error": "resume_ids is required."}),
            is_error=True,
        )
    try:
        resumes = await _run_sync(get_resumes_by_ids, resume_ids)
        trimmed = [
            {
                "resume_id": r["resume_id"],
                "candidate_name": r.get("candidate_name", "Unknown"),
                "semantic_text": (r.get("semantic_text") or "")[:1000],
                "metadata": r.get("metadata", {}),
            }
            for r in resumes
        ]
        return ToolResult(
            tool_call_id="",
            tool_name="get_candidate",
            output=json.dumps({"candidates": trimmed}, ensure_ascii=False),
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id="",
            tool_name="get_candidate",
            output=json.dumps({"error": str(exc)}),
            is_error=True,
        )


_TOOL_EXECUTORS = {
    "search_candidates": _exec_search_candidates,
    "score_candidates": _exec_score_candidates,
    "draft_schema": _exec_draft_schema,
    "save_schema": _exec_save_schema,
    "explain_score": _exec_explain_score,
    "get_active_schema": _exec_get_active_schema,
    "set_active_schema": _exec_set_active_schema,
    "list_schemas": _exec_list_schemas,
    "get_candidate": _exec_get_candidate,
}


async def execute_tool(tool_call: dict, ctx: "SessionContext") -> ToolResult:
    """Dispatch a single tool_call dict to the appropriate executor."""
    name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})
    tool_call_id = tool_call.get("id", "")

    executor = _TOOL_EXECUTORS.get(name)
    if executor is None:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=name,
            output=json.dumps({"error": f"Unknown tool: {name}"}),
            is_error=True,
        )

    result = await executor(arguments, ctx)
    result.tool_call_id = tool_call_id
    return result
