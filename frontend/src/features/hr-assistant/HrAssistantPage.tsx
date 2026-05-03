import { useCallback, useEffect, useRef, useState } from "react";
import {
  confirmPendingAction,
  createSession,
  deleteSession,
  getSession,
  listSessions,
  sendMessage,
} from "./hrAssistantApi";
import type {
  AgentEvent,
  ChatMessage,
  ChatSession,
  SchemaInfo,
  SchemaDraft,
  StreamState,
} from "./hrAssistantTypes";
import { ActiveSchemaPanel } from "./ActiveSchemaPanel";
import { ChatPanel } from "./ChatPanel";
import { SessionSidebar } from "./SessionSidebar";

const INITIAL_STREAM: StreamState = {
  isStreaming: false,
  pendingText: "",
  thinkingLabel: "",
  activeToolName: null,
  pendingConfirmation: null,
};

export function HrAssistantPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [stream, setStream] = useState<StreamState>(INITIAL_STREAM);
  const [activeSchema, setActiveSchema] = useState<SchemaInfo | null>(null);
  const [isSchemaLoading, setIsSchemaLoading] = useState(false);
  const [isSessionsLoading, setIsSessionsLoading] = useState(false);
  const [isConfirmSubmitting, setIsConfirmSubmitting] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);

  // Persist pending confirmation across renders without causing re-renders
  const pendingDraftRef = useRef<SchemaDraft | null>(null);

  // ── Load sessions list ──────────────────────────────────────
  const refreshSessions = useCallback(async () => {
    setIsSessionsLoading(true);
    try {
      const res = await listSessions();
      setSessions(res.sessions ?? []);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setIsSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions]);

  // ── Load session detail (messages + schema) ─────────────────
  const loadSession = useCallback(async (sessionId: string) => {
    try {
      const detail = await getSession(sessionId);
      setMessages(detail.messages ?? []);
      setActiveSessionId(sessionId);

      if (detail.active_schema_id) {
        setIsSchemaLoading(true);
        try {
          // Fetch schema from the existing schema endpoint
          const res = await fetch(
            `${import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000"}/api/scoring-schema/${detail.active_schema_id}`,
          );
          if (res.ok) setActiveSchema(await res.json());
          else setActiveSchema(null);
        } catch {
          setActiveSchema(null);
        } finally {
          setIsSchemaLoading(false);
        }
      } else {
        setActiveSchema(null);
      }
    } catch (err) {
      setPageError(`Failed to load session: ${err instanceof Error ? err.message : err}`);
    }
  }, []);

  // ── Create new session ──────────────────────────────────────
  const handleNewSession = useCallback(async () => {
    try {
      const res = await createSession();
      await refreshSessions();
      await loadSession(res.session_id);
      setStream(INITIAL_STREAM);
    } catch (err) {
      setPageError(`Failed to create session: ${err instanceof Error ? err.message : err}`);
    }
  }, [refreshSessions, loadSession]);

  // ── Delete session ──────────────────────────────────────────
  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      await refreshSessions();
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
        setActiveSchema(null);
        setStream(INITIAL_STREAM);
      }
    } catch (err) {
      setPageError(`Failed to delete session: ${err instanceof Error ? err.message : err}`);
    }
  }, [activeSessionId, refreshSessions]);

  // ── Handle SSE events ───────────────────────────────────────
  const handleEvent = useCallback((event: AgentEvent) => {
    switch (event.event) {
      case "thinking":
        setStream((s) => ({ ...s, thinkingLabel: String(event.data), activeToolName: null }));
        break;

      case "tool_start":
        setStream((s) => ({ ...s, activeToolName: event.data.tool }));
        break;

      case "tool_result":
        setStream((s) => ({ ...s, activeToolName: null }));
        // Tool results are persisted server-side; they'll appear when we reload messages
        break;

      case "confirmation_required": {
        const draft = event.data as SchemaDraft;
        pendingDraftRef.current = draft;
        setStream((s) => ({ ...s, pendingConfirmation: draft }));
        break;
      }

      case "text_chunk":
        setStream((s) => ({
          ...s,
          pendingText: s.pendingText + String(event.data),
          thinkingLabel: "",
        }));
        break;

      case "done":
        // Reload messages from DB to get the persisted view
        if (event.data.session_id) {
          void loadSession(event.data.session_id);
          void refreshSessions();
        }
        setStream((s) => ({
          ...s,
          isStreaming: false,
          pendingText: "",
          thinkingLabel: "",
          activeToolName: null,
        }));
        break;

      case "error":
        setPageError(event.data.message);
        setStream((s) => ({
          ...s,
          isStreaming: false,
          pendingText: "",
          thinkingLabel: "",
          activeToolName: null,
        }));
        break;
    }
  }, [loadSession, refreshSessions]);

  // ── Send message ────────────────────────────────────────────
  const handleSendMessage = useCallback(async (text: string) => {
    if (!activeSessionId) {
      // Create a new session on first message
      try {
        const res = await createSession();
        await refreshSessions();
        setActiveSessionId(res.session_id);
        setMessages([]);
        setStream({ ...INITIAL_STREAM, isStreaming: true, thinkingLabel: "Thinking…" });

        // Append user message optimistically
        const optimistic: ChatMessage = {
          message_id: `opt-${Date.now()}`,
          session_id: res.session_id,
          role: "user",
          content: text,
          created_at: new Date().toISOString(),
        };
        setMessages([optimistic]);

        await sendMessage(res.session_id, text, handleEvent, () => {});
      } catch (err) {
        setPageError(`Send failed: ${err instanceof Error ? err.message : err}`);
        setStream(INITIAL_STREAM);
      }
      return;
    }

    setStream({ ...INITIAL_STREAM, isStreaming: true, thinkingLabel: "Thinking…" });

    // Append user message optimistically
    const optimistic: ChatMessage = {
      message_id: `opt-${Date.now()}`,
      session_id: activeSessionId,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimistic]);

    try {
      await sendMessage(activeSessionId, text, handleEvent, () => {});
    } catch (err) {
      setPageError(`Send failed: ${err instanceof Error ? err.message : err}`);
      setStream(INITIAL_STREAM);
    }
  }, [activeSessionId, handleEvent, refreshSessions]);

  // ── Schema confirmation ─────────────────────────────────────
  const handleConfirm = useCallback(async () => {
    if (!activeSessionId) return;
    setIsConfirmSubmitting(true);
    try {
      await confirmPendingAction(activeSessionId, "confirm");
      pendingDraftRef.current = null;
      setStream((s) => ({ ...s, pendingConfirmation: null }));
      await loadSession(activeSessionId);
    } catch (err) {
      setPageError(`Confirmation failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setIsConfirmSubmitting(false);
    }
  }, [activeSessionId, loadSession]);

  const handleReject = useCallback(async () => {
    if (!activeSessionId) return;
    setIsConfirmSubmitting(true);
    try {
      await confirmPendingAction(activeSessionId, "reject");
      pendingDraftRef.current = null;
      setStream((s) => ({ ...s, pendingConfirmation: null }));
      await loadSession(activeSessionId);
    } catch (err) {
      setPageError(`Rejection failed: ${err instanceof Error ? err.message : err}`);
    } finally {
      setIsConfirmSubmitting(false);
    }
  }, [activeSessionId, loadSession]);

  return (
    <div className="hr-assistant-page">
      {pageError && (
        <div className="hr-page-error" role="alert">
          <span>{pageError}</span>
          <button type="button" onClick={() => setPageError(null)}>×</button>
        </div>
      )}

      <div className="hr-assistant-layout">
        <SessionSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          isLoadingSessions={isSessionsLoading}
          onSelectSession={loadSession}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
        />

        <ChatPanel
          messages={messages}
          streamState={stream}
          onSendMessage={handleSendMessage}
          onConfirm={handleConfirm}
          onReject={handleReject}
          isConfirmSubmitting={isConfirmSubmitting}
          activeSchemaName={activeSchema?.schema_name ?? null}
        />

        <ActiveSchemaPanel schema={activeSchema} isLoading={isSchemaLoading} />
      </div>
    </div>
  );
}
