import type { ChatSession } from "./hrAssistantTypes";

type SessionSidebarProps = {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isLoadingSessions: boolean;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
};

function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays === 0) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  isLoadingSessions,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}: SessionSidebarProps) {
  return (
    <aside className="hr-session-sidebar">
      <div className="hr-session-sidebar-header">
        <span className="eyebrow">Sessions</span>
        <button
          className="hr-new-session-btn"
          onClick={onNewSession}
          title="New conversation"
          type="button"
        >
          + New
        </button>
      </div>

      <div className="hr-session-list">
        {isLoadingSessions && <p className="muted" style={{ padding: "0.75rem", fontSize: "0.8rem" }}>Loading…</p>}
        {!isLoadingSessions && sessions.length === 0 && (
          <p className="muted" style={{ padding: "0.75rem", fontSize: "0.8rem" }}>No sessions yet</p>
        )}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`hr-session-item ${activeSessionId === s.session_id ? "active" : ""}`}
          >
            <button
              className="hr-session-item-btn"
              onClick={() => onSelectSession(s.session_id)}
              type="button"
              title={s.title ?? "Untitled"}
            >
              <span className="hr-session-title">{s.title || "Untitled"}</span>
              <span className="hr-session-date">{formatDate(s.updated_at)}</span>
            </button>
            <button
              className="hr-session-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(s.session_id);
              }}
              type="button"
              title="Delete session"
              aria-label="Delete session"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
