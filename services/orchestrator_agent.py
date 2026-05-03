"""
Orchestrator agent — multi-turn tool-calling loop with SSE event streaming.

Flow per turn:
  1. Load session context + message history
  2. Append user message
  3. Call LLM with tools (call_llm_with_tools)
  4. If tool calls: execute tools, emit events, append results, loop
  5. If text response: append assistant message, emit text_chunk + done
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional
from uuid import uuid4

from config import settings
from services.agent_tools import ALL_TOOLS, ToolResult, execute_tool
from services.llm_service import call_llm_with_tools
from services.session_service import (
    SessionContext,
    append_message,
    build_llm_messages,
    create_session,
    load_session,
    persist_session_state,
    update_session_title,
)
from utils.errors import LLMError
from utils.logger import get_logger

# Memory agent (imported lazily to avoid heavy startup cost)
def _memory_enabled() -> bool:
    return settings.MEMORY_AGENT_ENABLED and settings.ENABLE_DB_PERSISTENCE

logger = get_logger("orchestrator_agent")


SYSTEM_PROMPT = """You are an expert HR recruitment assistant for an AI resume ranking system.

Your job is to help HR professionals:
1. Search for candidates that match job requirements
2. Create and refine scoring schemas that evaluate candidates
3. Score and rank candidates using those schemas
4. Explain scoring results and provide insights

## How to work with schemas
- When an HR user describes their requirements, use draft_schema to generate a scoring schema
- Present the draft to the HR user for review — do NOT call save_schema unless the user confirms
- Once confirmed, call save_schema to persist the schema and activate it
- If hr_auto_mode is enabled, you may save directly without explicit confirmation

## Key workflow
1. Understand the HR's requirements through conversation
2. Use search_candidates to find matching resumes
3. Use draft_schema → confirm → save_schema to establish a scoring schema
4. Use score_candidates to rank the search results
5. Explain scores when asked

## Rules
- Always tell the HR user which schema is currently active before scoring
- When presenting search results, show candidate names and similarity scores
- When presenting score results, rank by score descending and highlight top candidates
- Be concise but helpful; ask clarifying questions when requirements are ambiguous
- Respond in the same language the HR user is using (Chinese or English)

## Available tools
- search_candidates: Find candidates by job description
- draft_schema: Generate a scoring schema draft for HR review
- save_schema: Save a confirmed schema to the database
- score_candidates: Score candidates using the active schema
- explain_score: Get detailed explanation for a specific candidate's score
- get_active_schema: Check which schema is currently active
- set_active_schema: Switch to a different existing schema
- list_schemas: List all available schemas
- get_candidate: Get detailed information about specific candidates
"""


async def run_agent_turn(
    session_id: str,
    user_message: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tool_rounds: int = 0,
) -> AsyncIterator[dict]:
    """
    Async generator yielding SSE event dicts for one HR conversation turn.

    Events:
      {"event": "thinking",              "data": "..."}
      {"event": "tool_start",            "data": {"tool": "...", "args": {...}}}
      {"event": "tool_result",           "data": {"tool": "...", "output": {...}}}
      {"event": "confirmation_required", "data": {"schema_name": "...", ...}}
      {"event": "text_chunk",            "data": "...partial text..."}
      {"event": "done",                  "data": {"session_id": "...", "message_id": "..."}}
      {"event": "error",                 "data": {"message": "..."}}
    """
    if max_tool_rounds <= 0:
        max_tool_rounds = settings.AGENT_MAX_TOOL_ROUNDS

    try:
        ctx = load_session(session_id)
        if ctx is None:
            yield {"event": "error", "data": {"message": f"Session {session_id} not found."}}
            return

        effective_provider = provider or ctx.provider or settings.AGENT_LLM_PROVIDER
        effective_model = model or ctx.model or settings.AGENT_LLM_MODEL

        # Append user message to DB
        append_message(session_id, "user", user_message)

        # Auto-generate session title from first user message
        messages_so_far = build_llm_messages(session_id, limit=2)
        if len(messages_so_far) == 1:
            title = user_message[:60].replace("\n", " ").strip()
            update_session_title(session_id, title)

        # ── Retrieve long-term memories relevant to this message ──────────
        memory_context = ""
        if _memory_enabled():
            try:
                from services.memory_agent import (
                    build_memory_context_block,
                    retrieve_relevant_memories,
                )
                loop = asyncio.get_event_loop()
                memories = await loop.run_in_executor(
                    None,
                    lambda: retrieve_relevant_memories(
                        user_message, current_session_id=session_id
                    ),
                )
                memory_context = build_memory_context_block(memories)
            except Exception as exc:
                logger.warning("Memory retrieval skipped: %s", exc)

        async for event in _tool_calling_loop(
            ctx=ctx,
            provider=effective_provider,
            model=effective_model,
            max_rounds=max_tool_rounds,
            memory_context=memory_context,
        ):
            yield event

        # ── Async post-turn: extract memories in background ───────────────
        if _memory_enabled():
            asyncio.get_event_loop().run_in_executor(
                None, _extract_memories_background, session_id
            )

    except Exception as exc:
        logger.exception("Unhandled error in run_agent_turn: %s", exc)
        yield {"event": "error", "data": {"message": f"Internal agent error: {exc}"}}


def _extract_memories_background(session_id: str) -> None:
    """Run memory extraction synchronously (called from run_in_executor)."""
    try:
        from services.memory_agent import extract_session_memories
        extract_session_memories(session_id)
    except Exception as exc:
        logger.warning("Background memory extraction failed for %s: %s", session_id, exc)


async def _tool_calling_loop(
    ctx: SessionContext,
    provider: str,
    model: Optional[str],
    max_rounds: int,
    memory_context: str = "",
) -> AsyncIterator[dict]:
    loop = asyncio.get_event_loop()
    final_message_id: Optional[str] = None

    for round_num in range(max_rounds):
        messages = build_llm_messages(ctx.session_id, limit=60)

        yield {"event": "thinking", "data": f"Round {round_num + 1}: calling LLM..."}

        system = SYSTEM_PROMPT
        if memory_context:
            system = f"{SYSTEM_PROMPT}\n\n{memory_context}"

        try:
            text_response, tool_calls, _usage = await loop.run_in_executor(
                None,
                lambda: call_llm_with_tools(
                    messages,
                    ALL_TOOLS,
                    provider=provider,
                    model=model,
                    system=system,
                ),
            )
        except LLMError as exc:
            yield {"event": "error", "data": {"message": f"LLM error: {exc}"}}
            return

        # ── LLM chose to respond in text ──────────────────────────────────
        if text_response:
            final_message_id = append_message(ctx.session_id, "assistant", text_response)
            persist_session_state(ctx)

            # Stream text in chunks for a smoother UX
            chunk_size = 80
            for i in range(0, len(text_response), chunk_size):
                yield {"event": "text_chunk", "data": text_response[i: i + chunk_size]}

            yield {
                "event": "done",
                "data": {"session_id": ctx.session_id, "message_id": final_message_id},
            }
            return

        # ── LLM chose to call tools ────────────────────────────────────────
        if not tool_calls:
            # No text, no tools — force a stop
            break

        # Persist the assistant turn that contains tool_calls
        assistant_content = ""
        append_message(
            ctx.session_id,
            "assistant",
            assistant_content,
            tool_calls=tool_calls,
        )

        # Execute each tool call
        confirmation_emitted = False
        tool_results: list[ToolResult] = []

        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("arguments", {})

            yield {"event": "tool_start", "data": {"tool": tool_name, "args": tool_args}}

            result = await execute_tool(tc, ctx)
            tool_results.append(result)

            # Parse output for the event (trim large payloads)
            try:
                output_obj = json.loads(result.output)
            except Exception:
                output_obj = {"raw": result.output[:500]}

            yield {"event": "tool_result", "data": {"tool": tool_name, "output": output_obj}}

            # Emit confirmation gate if schema draft needs HR approval
            if result.requires_confirmation and result.confirmation_payload and not confirmation_emitted:
                yield {"event": "confirmation_required", "data": result.confirmation_payload}
                confirmation_emitted = True

            # Persist tool result message
            append_message(
                ctx.session_id,
                "tool",
                result.output,
                tool_name=tool_name,
                tool_call_id=tc.get("id", ""),
            )

        # If confirmation was required, pause the loop — let the next HR turn handle it.
        # The loop continues normally when hr_auto_mode is True (save_schema already called).
        if confirmation_emitted:
            # Emit a synthetic assistant prompt asking HR to confirm
            confirm_msg = (
                "I've prepared a scoring schema draft. Please review it above and reply "
                "'confirm' to save it, or let me know what changes you'd like."
            )
            final_message_id = append_message(ctx.session_id, "assistant", confirm_msg)
            for i in range(0, len(confirm_msg), 80):
                yield {"event": "text_chunk", "data": confirm_msg[i: i + 80]}
            yield {
                "event": "done",
                "data": {"session_id": ctx.session_id, "message_id": final_message_id},
            }
            return

    # Exceeded max_rounds without a text response
    fallback = "I've completed the requested operations. Let me know if you need anything else."
    final_message_id = append_message(ctx.session_id, "assistant", fallback)
    persist_session_state(ctx)
    yield {"event": "text_chunk", "data": fallback}
    yield {
        "event": "done",
        "data": {"session_id": ctx.session_id, "message_id": final_message_id},
    }


async def confirm_pending_schema(session_id: str, action: str) -> dict:
    """
    Handle HR confirmation or rejection of a pending schema draft.
    action: "confirm" | "reject"
    """
    ctx = load_session(session_id)
    if ctx is None:
        return {"error": f"Session {session_id} not found."}

    if not ctx.pending_confirmation:
        return {"error": "No pending schema confirmation for this session."}

    if action == "reject":
        ctx.pending_confirmation = None
        persist_session_state(ctx)
        append_message(session_id, "user", "Schema draft rejected.")
        append_message(session_id, "assistant", "Schema draft discarded. Let me know how you'd like to adjust the requirements.")
        return {"status": "rejected"}

    if action == "confirm":
        from services.agent_tools import _exec_save_schema
        draft = ctx.pending_confirmation
        result = await _exec_save_schema(
            {"schema_name": draft["schema_name"], "rules_text": draft["rules_text"]},
            ctx,
        )
        try:
            output = json.loads(result.output)
        except Exception:
            output = {}
        if result.is_error:
            return {"error": output.get("error", "Save failed.")}
        append_message(session_id, "user", "Schema confirmed.")
        append_message(
            session_id,
            "assistant",
            f"Schema '{draft['schema_name']}' has been saved and is now active. Ready to score candidates!",
        )
        return {"status": "confirmed", **output}

    return {"error": f"Unknown action: {action}. Use 'confirm' or 'reject'."}
