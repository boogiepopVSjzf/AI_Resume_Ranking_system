"""
Session management for multi-turn HR conversation agent.

SessionContext holds in-process working memory for one request turn.
Persistence (chat_sessions, chat_messages tables) is handled via postgres_store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from config import settings
from storage.postgres_store import (
    append_chat_message,
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    list_chat_sessions,
    load_chat_session,
    save_chat_session,
)
from utils.errors import DatabaseError
from utils.logger import get_logger

logger = get_logger("session_service")


@dataclass
class SessionContext:
    """Working memory for the current request turn. Rebuilt from DB each request."""

    session_id: str
    active_schema_id: Optional[str] = None
    hr_auto_mode: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None

    # In-process working memory (not persisted directly — stored via append_chat_message)
    recent_search_result_ids: list[str] = field(default_factory=list)
    recent_score_results: list[dict] = field(default_factory=list)
    # Pending schema draft awaiting HR confirmation
    pending_confirmation: Optional[dict] = None


def create_session(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    hr_auto_mode: bool = False,
    title: Optional[str] = None,
) -> SessionContext:
    session_id = uuid4().hex
    create_chat_session(
        session_id=session_id,
        title=title or "New Conversation",
        hr_auto_mode=hr_auto_mode,
        provider=provider or settings.AGENT_LLM_PROVIDER,
        model=model or settings.AGENT_LLM_MODEL,
    )
    return SessionContext(
        session_id=session_id,
        hr_auto_mode=hr_auto_mode or settings.AGENT_HR_AUTO_MODE,
        provider=provider or settings.AGENT_LLM_PROVIDER,
        model=model or settings.AGENT_LLM_MODEL,
    )


def load_session(session_id: str) -> Optional[SessionContext]:
    row = load_chat_session(session_id)
    if row is None:
        return None
    return SessionContext(
        session_id=row["session_id"],
        active_schema_id=row.get("active_schema_id"),
        hr_auto_mode=row.get("hr_auto_mode", False),
        provider=row.get("provider"),
        model=row.get("model"),
    )


def persist_session_state(ctx: SessionContext) -> None:
    """Flush mutable session fields back to the DB."""
    save_chat_session(
        session_id=ctx.session_id,
        active_schema_id=ctx.active_schema_id,
        hr_auto_mode=ctx.hr_auto_mode,
    )


def update_session_title(session_id: str, title: str) -> None:
    save_chat_session(session_id=session_id, title=title)


def update_active_schema(ctx: SessionContext, schema_id: str) -> None:
    ctx.active_schema_id = schema_id
    save_chat_session(session_id=ctx.session_id, active_schema_id=schema_id)


def get_session_messages(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    return get_chat_messages(session_id, limit=limit)


def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    tool_name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[list] = None,
) -> str:
    message_id = uuid4().hex
    append_chat_message(
        message_id=message_id,
        session_id=session_id,
        role=role,
        content=content,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_calls_json=tool_calls,
    )
    return message_id


def build_llm_messages(session_id: str, limit: int = 40) -> list[dict]:
    """
    Reconstruct the LLM message history from persisted chat_messages.
    Converts tool rows back into the provider-agnostic dict format used by
    call_llm_with_tools.
    """
    rows = get_chat_messages(session_id, limit=limit)
    messages: list[dict] = []
    for row in rows:
        role = row["role"]
        if role == "system":
            continue  # system prompt is injected separately
        content = row["content"]
        if role == "tool":
            messages.append({
                "role": "tool",
                "tool_name": row.get("tool_name") or "tool",
                "tool_call_id": row.get("tool_call_id") or "",
                "content": content,
            })
        elif role == "assistant" and row.get("tool_calls_json"):
            tool_calls = row["tool_calls_json"]
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except Exception:
                    tool_calls = []
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            })
        else:
            messages.append({"role": role, "content": content})
    return messages


def get_sessions_list(limit: int = 20) -> list[dict[str, Any]]:
    return list_chat_sessions(limit=limit)


def remove_session(session_id: str) -> None:
    delete_chat_session(session_id)


def find_or_create_session(
    initial_message: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    hr_auto_mode: bool = False,
) -> tuple[str, bool]:
    """
    Try to find a semantically similar recent session for auto-continuation.
    Returns (session_id, is_resumed).

    If no match is found, a new session is created and (new_session_id, False) returned.
    """
    from services.memory_agent import find_continuation_session
    matched_id = find_continuation_session(initial_message)
    if matched_id:
        return matched_id, True

    ctx = create_session(
        provider=provider,
        model=model,
        hr_auto_mode=hr_auto_mode,
    )
    return ctx.session_id, False
