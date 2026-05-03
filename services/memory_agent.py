"""
Memory Agent — hermes-agent style long-term memory management.

Responsibilities:
1. Extract structured memories from a completed session (post-session)
2. Retrieve relevant memories before a new turn (pre-turn)
3. Find semantically similar past sessions for auto-continuation

Memory types stored in agent_memories table:
  hr_preference       — HR's stated evaluation preferences
  session_summary     — Distilled summary of an entire session (with embedding)
  search_result_cache — Cached search results keyed by query hash
  schema_draft_history— Schemas discussed / saved in a session
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional
from uuid import uuid4

from config import settings
from services.embedding_service import embed_text
from services.llm_service import call_llm
from storage.postgres_store import (
    cache_search_results,
    find_similar_session,
    get_cached_search_results,
    get_chat_messages,
    get_memories_by_session,
    invalidate_session_search_cache,
    retrieve_memories_by_embedding,
    store_memory,
)
from utils.errors import LLMError, LLMParseError
from utils.llm_json import extract_json
from utils.logger import get_logger

logger = get_logger("memory_agent")


# ─────────────────────────────────────────────────────────────
# Memory extraction (called at end of session)
# ─────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT_TEMPLATE = """You are a memory extraction assistant for an HR recruitment system.

Analyze the conversation below and extract structured information.

Respond with a JSON object matching this exact schema:
{{
  "session_summary": "<1-3 sentence summary of what was accomplished in this session>",
  "hr_preferences": ["<preference 1>", "<preference 2>", ...],
  "schemas_discussed": [
    {{"name": "<schema name>", "description": "<brief description>", "saved": true/false}}
  ],
  "candidates_reviewed": [
    {{"resume_id": "<id>", "verdict": "<excellent/good/qualified/bad/unknown>"}}
  ]
}}

Rules:
- hr_preferences: explicit criteria the HR mentioned (e.g. "prefers 5+ years", "values domain experience")
- If no schemas were discussed, use an empty array
- If no candidates were reviewed, use an empty array
- Output valid JSON only, no markdown fences

Conversation:
{conversation}
"""


def _format_conversation_for_extraction(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            continue
        if role == "tool":
            tool_name = msg.get("tool_name", "tool")
            try:
                output = json.loads(content)
                content = json.dumps(output, ensure_ascii=False)[:400]
            except Exception:
                content = content[:400]
            lines.append(f"[Tool: {tool_name}] {content}")
        elif role == "assistant":
            if msg.get("tool_calls_json"):
                continue  # skip raw tool-call turns
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"HR: {content}")
    return "\n".join(lines)


def extract_session_memories(session_id: str) -> dict:
    """
    Read all messages for a session, call LLM to extract structured memories,
    then persist hr_preference, session_summary, and schema_draft_history entries.

    Returns the extracted data dict (for logging / testing).
    """
    messages = get_chat_messages(session_id, limit=200)
    if not messages:
        return {}

    conversation_text = _format_conversation_for_extraction(messages)
    if len(conversation_text.strip()) < 50:
        return {}

    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(conversation=conversation_text[:6000])

    try:
        raw, _ = call_llm(
            prompt,
            provider=settings.AGENT_LLM_PROVIDER,
            model=settings.AGENT_LLM_MODEL,
        )
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise LLMParseError("Memory extraction returned non-dict JSON")
    except (LLMError, LLMParseError) as exc:
        logger.warning("Memory extraction failed for session %s: %s", session_id, exc)
        return {}

    # ── Persist session summary ──────────────────────────────────────────
    summary_text = (data.get("session_summary") or "").strip()
    if summary_text:
        try:
            summary_embedding = embed_text(summary_text)
        except Exception:
            summary_embedding = None
        store_memory(
            memory_id=f"sum_{session_id}",
            session_id=session_id,
            memory_type="session_summary",
            content=summary_text,
            content_json={"session_id": session_id},
            embedding=summary_embedding,
        )

    # ── Persist HR preferences ───────────────────────────────────────────
    for i, pref in enumerate(data.get("hr_preferences") or []):
        pref_text = (pref or "").strip()
        if not pref_text:
            continue
        store_memory(
            memory_id=f"pref_{session_id}_{i}",
            session_id=session_id,
            memory_type="hr_preference",
            content=pref_text,
        )

    # ── Persist schema draft history ─────────────────────────────────────
    for schema in data.get("schemas_discussed") or []:
        name = (schema.get("name") or "").strip()
        if not name:
            continue
        store_memory(
            memory_id=f"schema_{session_id}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            session_id=session_id,
            memory_type="schema_draft_history",
            content=f"Schema '{name}': {schema.get('description', '')}",
            content_json=schema,
        )

    logger.info(
        "Extracted memories for session %s: summary=%s, prefs=%d, schemas=%d",
        session_id,
        bool(summary_text),
        len(data.get("hr_preferences") or []),
        len(data.get("schemas_discussed") or []),
    )
    return data


# ─────────────────────────────────────────────────────────────
# Memory retrieval (called before each agent turn)
# ─────────────────────────────────────────────────────────────

def retrieve_relevant_memories(
    query_text: str,
    *,
    current_session_id: Optional[str] = None,
    limit: int = 0,
) -> list[dict]:
    """
    Embed the query text and return the most relevant memories across all sessions.
    Only returns hr_preference and session_summary types (not transient caches).
    """
    if limit <= 0:
        limit = settings.MEMORY_RETRIEVAL_LIMIT
    if not query_text.strip():
        return []

    try:
        query_embedding = embed_text(query_text)
    except Exception as exc:
        logger.warning("Failed to embed query for memory retrieval: %s", exc)
        return []

    try:
        memories = retrieve_memories_by_embedding(
            query_embedding,
            memory_types=["hr_preference", "session_summary"],
            limit=limit,
        )
        # Don't show the summary for the current session (it would be self-referential)
        if current_session_id:
            memories = [
                m for m in memories
                if not (m["memory_type"] == "session_summary" and m["session_id"] == current_session_id)
            ]
        return memories
    except Exception as exc:
        logger.warning("Failed to retrieve memories: %s", exc)
        return []


def build_memory_context_block(memories: list[dict]) -> str:
    """
    Format retrieved memories as a <memory> block to prepend to the system prompt.
    Returns empty string if no memories.
    """
    if not memories:
        return ""

    prefs = [m["content"] for m in memories if m["memory_type"] == "hr_preference"]
    summaries = [m["content"] for m in memories if m["memory_type"] == "session_summary"]

    lines = ["<memory>"]
    if prefs:
        lines.append("HR preferences from past sessions:")
        for p in prefs:
            lines.append(f"  - {p}")
    if summaries:
        lines.append("Relevant past sessions:")
        for s in summaries[:3]:
            lines.append(f"  - {s}")
    lines.append("</memory>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Session auto-continuation (semantic matching)
# ─────────────────────────────────────────────────────────────

def find_continuation_session(initial_message: str) -> Optional[str]:
    """
    Embed the initial message and search for a semantically similar recent session.
    Returns the session_id if similarity >= threshold, else None.
    """
    try:
        embedding = embed_text(initial_message)
    except Exception as exc:
        logger.warning("Embed failed for session continuation: %s", exc)
        return None

    try:
        match = find_similar_session(
            embedding,
            similarity_threshold=settings.SESSION_CONTINUATION_THRESHOLD,
            window_days=settings.SESSION_CONTINUATION_WINDOW_DAYS,
        )
        if match:
            logger.info(
                "Auto-continuing session %s (similarity=%.3f)",
                match["session_id"],
                match["similarity"],
            )
            return match["session_id"]
    except Exception as exc:
        logger.warning("Session continuation search failed: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────
# Search result cache helpers
# ─────────────────────────────────────────────────────────────

def make_search_cache_key(job_description: str, filter_mode: str, top_k: int) -> str:
    payload = json.dumps(
        {"jd": job_description.strip(), "mode": filter_mode, "top_k": top_k},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get_search_cache(session_id: str, cache_key: str) -> Optional[dict]:
    try:
        return get_cached_search_results(session_id, cache_key)
    except Exception:
        return None


def set_search_cache(session_id: str, cache_key: str, results: dict) -> None:
    try:
        cache_search_results(
            session_id, cache_key, results,
            ttl_hours=settings.SEARCH_CACHE_TTL_HOURS,
        )
    except Exception as exc:
        logger.warning("Failed to cache search results: %s", exc)


def invalidate_search_cache(session_id: str) -> None:
    """Called when a new schema is saved — stale search scores are no longer valid."""
    try:
        invalidate_session_search_cache(session_id)
    except Exception as exc:
        logger.warning("Failed to invalidate search cache: %s", exc)
