import { API_BASE_URL } from "../../lib/constants";
import type {
  AgentEvent,
  ChatSession,
  SchemaInfo,
  SessionDetail,
  SchemaDraft,
} from "./hrAssistantTypes";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Session management ────────────────────────────────────────

export async function createSession(options?: {
  provider?: string;
  model?: string;
  hr_auto_mode?: boolean;
  title?: string;
}): Promise<{ session_id: string; provider: string | null; model: string | null; hr_auto_mode: boolean }> {
  return apiFetch("/api/chat/session", {
    method: "POST",
    body: JSON.stringify(options ?? {}),
  });
}

export async function listSessions(): Promise<{ sessions: ChatSession[] }> {
  return apiFetch("/api/chat/sessions");
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return apiFetch(`/api/chat/session/${sessionId}`);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiFetch(`/api/chat/session/${sessionId}`, { method: "DELETE" });
}

export async function confirmPendingAction(
  sessionId: string,
  action: "confirm" | "reject",
): Promise<Record<string, unknown>> {
  return apiFetch(`/api/chat/session/${sessionId}/confirm`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}

export async function getSchemaInfo(schemaId: string): Promise<SchemaInfo | null> {
  try {
    return await apiFetch(`/api/scoring-schema/${schemaId}`);
  } catch {
    return null;
  }
}

// ── SSE streaming helpers ─────────────────────────────────────

/**
 * Parse a raw SSE text stream into AgentEvent objects.
 * Calls onEvent for each complete event frame.
 * Calls onDone when the stream ends.
 */
export async function streamMessages(
  url: string,
  body: Record<string, unknown>,
  onEvent: (event: AgentEvent) => void,
  onDone: () => void,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const line = frame.trim();
        if (!line || line === "data: [DONE]") continue;
        if (line.startsWith("data: ")) {
          const raw = line.slice(6).trim();
          try {
            const parsed = JSON.parse(raw) as AgentEvent;
            onEvent(parsed);
          } catch {
            // ignore malformed frames
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
    onDone();
  }
}

/** Send a message to an existing session via SSE. */
export function sendMessage(
  sessionId: string,
  message: string,
  onEvent: (event: AgentEvent) => void,
  onDone: () => void,
): Promise<void> {
  return streamMessages(
    `/api/chat/${sessionId}/message`,
    { message },
    onEvent,
    onDone,
  );
}

/** Fetch a short-lived presigned S3 URL for a resume file. */
export async function getResumeUrl(resumeId: string): Promise<string | null> {
  try {
    const res = await apiFetch<{ url: string }>(`/api/resume/${resumeId}/url`);
    return res.url;
  } catch {
    return null;
  }
}

/** Send a message without a session (auto-create/resume). */
export function sendMessageAuto(
  message: string,
  onEvent: (event: AgentEvent) => void,
  onDone: () => void,
): Promise<void> {
  return streamMessages("/api/chat/message", { message }, onEvent, onDone);
}
