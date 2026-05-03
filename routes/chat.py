"""
Chat API routes — multi-turn HR conversation agent with SSE streaming.

Endpoints:
  POST   /api/chat/session                      — create new session
  GET    /api/chat/sessions                     — list recent sessions
  GET    /api/chat/session/{session_id}         — get session + message history
  DELETE /api/chat/session/{session_id}         — delete session
  POST   /api/chat/session/{session_id}/confirm — confirm/reject pending schema draft
  POST   /api/chat/{session_id}/message         — main SSE chat endpoint
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import settings
from services.orchestrator_agent import confirm_pending_schema, run_agent_turn
from services.session_service import (
    create_session,
    find_or_create_session,
    get_session_messages,
    get_sessions_list,
    load_session,
    remove_session,
)
from utils.errors import DatabaseError
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("chat")


# ─────────────────────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    hr_auto_mode: bool = False
    title: Optional[str] = None


class SendMessageRequest(BaseModel):
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None


class ConfirmActionRequest(BaseModel):
    action: str  # "confirm" | "reject"


class AutoMessageRequest(BaseModel):
    """Used by POST /api/chat/message — no session_id required."""
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    hr_auto_mode: bool = False


# ─────────────────────────────────────────────────────────────
# Session management endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/api/chat/session")
async def create_chat_session(req: CreateSessionRequest) -> JSONResponse:
    """Create a new conversation session."""
    if not settings.ENABLE_DB_PERSISTENCE:
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable the chat agent.",
        )
    try:
        ctx = create_session(
            provider=req.provider,
            model=req.model,
            hr_auto_mode=req.hr_auto_mode,
            title=req.title,
        )
        return JSONResponse({
            "session_id": ctx.session_id,
            "provider": ctx.provider,
            "model": ctx.model,
            "hr_auto_mode": ctx.hr_auto_mode,
        })
    except DatabaseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/chat/sessions")
async def list_chat_sessions() -> JSONResponse:
    """Return the 20 most recent sessions."""
    if not settings.ENABLE_DB_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Database not configured.")
    try:
        sessions = get_sessions_list(limit=20)
        return JSONResponse({"sessions": sessions})
    except DatabaseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/chat/session/{session_id}")
async def get_chat_session(session_id: str) -> JSONResponse:
    """Return session metadata and full message history."""
    if not settings.ENABLE_DB_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Database not configured.")
    try:
        ctx = load_session(session_id)
        if ctx is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
        messages = get_session_messages(session_id, limit=200)
        return JSONResponse({
            "session_id": ctx.session_id,
            "active_schema_id": ctx.active_schema_id,
            "hr_auto_mode": ctx.hr_auto_mode,
            "provider": ctx.provider,
            "model": ctx.model,
            "messages": messages,
        })
    except HTTPException:
        raise
    except DatabaseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/api/chat/session/{session_id}")
async def delete_chat_session(session_id: str) -> JSONResponse:
    """Delete a session and all its messages."""
    if not settings.ENABLE_DB_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Database not configured.")
    try:
        remove_session(session_id)
        return JSONResponse({"status": "deleted", "session_id": session_id})
    except DatabaseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/api/chat/session/{session_id}/confirm")
async def confirm_schema_action(session_id: str, req: ConfirmActionRequest) -> JSONResponse:
    """Confirm or reject a pending schema draft."""
    if not settings.ENABLE_DB_PERSISTENCE:
        raise HTTPException(status_code=503, detail="Database not configured.")
    if req.action not in ("confirm", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'confirm' or 'reject'")
    try:
        result = await confirm_pending_schema(session_id, req.action)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(result)
    except HTTPException:
        raise
    except DatabaseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# Auto-session SSE endpoint (session auto-created or resumed)
# ─────────────────────────────────────────────────────────────

@router.post("/api/chat/message")
async def send_message_auto_session(req: AutoMessageRequest) -> StreamingResponse:
    """
    SSE endpoint that automatically creates or resumes a session.

    First, performs semantic matching against recent session summaries.
    If a similar session is found (similarity >= SESSION_CONTINUATION_THRESHOLD),
    it resumes that session. Otherwise creates a new one.

    The resolved session_id is returned in the SSE 'done' event.
    """
    if not settings.ENABLE_DB_PERSISTENCE:
        async def db_error_stream():
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Database not configured.'}})}\n\n"
        return StreamingResponse(db_error_stream(), media_type="text/event-stream")

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    async def event_stream():
        try:
            loop = __import__("asyncio").get_event_loop()
            session_id, is_resumed = await loop.run_in_executor(
                None,
                lambda: find_or_create_session(
                    req.message.strip(),
                    provider=req.provider,
                    model=req.model,
                    hr_auto_mode=req.hr_auto_mode,
                ),
            )
            # Inform the frontend whether we resumed or created
            yield f"data: {json.dumps({'event': 'session_resolved', 'data': {'session_id': session_id, 'resumed': is_resumed}}, ensure_ascii=False)}\n\n"

            async for event in run_agent_turn(
                session_id,
                req.message.strip(),
                provider=req.provider,
                model=req.model,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Unexpected error in auto-session stream: %s", exc)
            err = {"event": "error", "data": {"message": str(exc)}}
            yield f"data: {json.dumps(err)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────
# Main SSE chat endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/api/chat/{session_id}/message")
async def send_message(session_id: str, req: SendMessageRequest) -> StreamingResponse:
    """
    Main SSE endpoint. Sends a user message and streams agent events.

    Each SSE frame has the format:
        data: {"event": "<type>", "data": <payload>}\\n\\n

    Event types: thinking, tool_start, tool_result, confirmation_required,
                 text_chunk, done, error
    """
    if not settings.ENABLE_DB_PERSISTENCE:
        async def db_error_stream():
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Database not configured.'}})}\n\n"
        return StreamingResponse(db_error_stream(), media_type="text/event-stream")

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    async def event_stream():
        try:
            async for event in run_agent_turn(
                session_id,
                req.message.strip(),
                provider=req.provider,
                model=req.model,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Unexpected error in event stream: %s", exc)
            err = {"event": "error", "data": {"message": str(exc)}}
            yield f"data: {json.dumps(err)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
