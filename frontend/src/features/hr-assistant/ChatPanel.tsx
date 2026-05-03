import { useEffect, useRef, useState } from "react";
import type { ChatMessage, SchemaDraft, StreamState } from "./hrAssistantTypes";
import { getResumeUrl } from "./hrAssistantApi";

type ChatPanelProps = {
  messages: ChatMessage[];
  streamState: StreamState;
  onSendMessage: (text: string) => void;
  onConfirm: () => void;
  onReject: () => void;
  isConfirmSubmitting: boolean;
  activeSchemaName: string | null;
};

type SearchResult = { resume_id: string; name: string; similarity_score: number };

function CandidateCard({ result }: { result: SearchResult }) {
  const [loading, setLoading] = useState(false);

  async function handleViewResume() {
    setLoading(true);
    const url = await getResumeUrl(result.resume_id);
    setLoading(false);
    if (url) {
      window.open(url, "_blank", "noopener,noreferrer");
    } else {
      alert("Resume file not available");
    }
  }

  return (
    <div className="hr-candidate-card">
      <div className="hr-candidate-info">
        <span className="hr-candidate-name">{result.name}</span>
        <span className="hr-candidate-score">
          {(result.similarity_score * 100).toFixed(1)}% match
        </span>
      </div>
      <button
        className="hr-candidate-view-btn"
        onClick={handleViewResume}
        disabled={loading}
        type="button"
      >
        {loading ? "…" : "View Resume"}
      </button>
    </div>
  );
}

function ToolCallBubble({ msg }: { msg: ChatMessage }) {
  const [open, setOpen] = useState(false);

  // Render search_candidates results as candidate cards
  if (msg.tool_name === "search_candidates") {
    let parsed: { results?: SearchResult[] } = {};
    try { parsed = JSON.parse(msg.content); } catch { /* ignore */ }
    const results = parsed.results ?? [];
    if (results.length > 0) {
      return (
        <div className="hr-msg-tool hr-msg-tool-candidates">
          <div className="hr-msg-tool-header">
            <span className="hr-tool-icon">⚙</span>
            <span className="hr-tool-name">Found {results.length} candidate{results.length !== 1 ? "s" : ""}</span>
          </div>
          <div className="hr-candidate-list">
            {results.map((r) => (
              <CandidateCard key={r.resume_id} result={r} />
            ))}
          </div>
        </div>
      );
    }
  }

  // Generic collapsible for all other tools
  let output: unknown;
  try {
    output = JSON.parse(msg.content);
  } catch {
    output = msg.content;
  }
  const preview = typeof output === "object" && output !== null
    ? JSON.stringify(output, null, 2).slice(0, 300)
    : String(output).slice(0, 300);

  return (
    <div className="hr-msg-tool">
      <button
        className="hr-msg-tool-toggle"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <span className="hr-tool-icon">⚙</span>
        <span className="hr-tool-name">{msg.tool_name ?? "tool"}</span>
        <span className="hr-tool-chevron">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <pre className="hr-msg-tool-body">{preview}{preview.length >= 300 ? "\n…" : ""}</pre>
      )}
    </div>
  );
}

function AssistantBubble({ content }: { content: string }) {
  // Very minimal markdown: **bold**, newlines → <br>
  const html = content
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");
  return (
    <div
      className="hr-msg-assistant"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function ChatPanel({
  messages,
  streamState,
  onSendMessage,
  onConfirm,
  onReject,
  isConfirmSubmitting,
  activeSchemaName,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamState.pendingText, streamState.thinkingLabel]);

  function handleSend() {
    const text = input.trim();
    if (!text || streamState.isStreaming) return;
    setInput("");
    onSendMessage(text);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const visibleMessages = messages.filter(
    (m) => m.role !== "system" && !(m.role === "assistant" && m.tool_calls_json?.length),
  );

  return (
    <div className="hr-chat-area">
      {/* Active schema badge */}
      {activeSchemaName && (
        <div className="hr-schema-badge">
          <span className="hr-schema-badge-dot" />
          <span>Schema: <strong>{activeSchemaName}</strong></span>
        </div>
      )}

      {/* Message list */}
      <div className="hr-messages">
        {visibleMessages.length === 0 && !streamState.isStreaming && (
          <div className="hr-empty-state">
            <p>Start a conversation to search candidates, create schemas, or score resumes.</p>
            <div className="hr-suggestions">
              {[
                "我需要找一个有5年Python经验的数据科学家",
                "Help me create a scoring schema for a backend engineer role",
                "Show me available scoring schemas",
              ].map((s) => (
                <button
                  key={s}
                  className="hr-suggestion-btn"
                  onClick={() => onSendMessage(s)}
                  type="button"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {visibleMessages.map((msg) => {
          if (msg.role === "tool") return <ToolCallBubble key={msg.message_id} msg={msg} />;
          if (msg.role === "user") {
            return (
              <div key={msg.message_id} className="hr-msg-row hr-msg-row-user">
                <div className="hr-msg-user">{msg.content}</div>
              </div>
            );
          }
          if (msg.role === "assistant") {
            return (
              <div key={msg.message_id} className="hr-msg-row hr-msg-row-assistant">
                <AssistantBubble content={msg.content} />
              </div>
            );
          }
          return null;
        })}

        {/* Streaming indicators */}
        {streamState.isStreaming && streamState.thinkingLabel && !streamState.pendingText && (
          <div className="hr-msg-row hr-msg-row-assistant">
            <div className="hr-msg-thinking">
              {streamState.activeToolName
                ? `⚙ Running ${streamState.activeToolName}…`
                : streamState.thinkingLabel}
            </div>
          </div>
        )}

        {/* Streaming text in progress */}
        {streamState.pendingText && (
          <div className="hr-msg-row hr-msg-row-assistant">
            <AssistantBubble content={streamState.pendingText} />
          </div>
        )}

        {/* Schema confirmation inline */}
        {streamState.pendingConfirmation && (
          <div className="hr-msg-row hr-msg-row-assistant">
            <div className="hr-inline-confirm">
              <p className="eyebrow" style={{ marginBottom: "0.5rem" }}>Schema Draft — Review Required</p>
              <strong>{streamState.pendingConfirmation.schema_name}</strong>
              <p style={{ marginTop: "0.4rem", fontSize: "0.85rem", color: "var(--muted)" }}>
                {streamState.pendingConfirmation.summary}
              </p>
              <div className="hr-confirm-rule-list">
                {Object.entries(streamState.pendingConfirmation.rules_json ?? {}).map(([key, rule]) => (
                  <div key={key} className="hr-confirm-rule-row">
                    <span className="hr-confirm-rule-key">{key}</span>
                    <span className="hr-confirm-rule-weight">{Math.round((rule.weight ?? 0) * 100)}%</span>
                    <span className="hr-confirm-rule-text">{rule.rule_text}</span>
                  </div>
                ))}
              </div>
              <div className="hr-confirm-actions" style={{ marginTop: "0.75rem" }}>
                <button className="btn-secondary" onClick={onReject} disabled={isConfirmSubmitting} type="button">
                  Reject
                </button>
                <button className="btn-primary" onClick={onConfirm} disabled={isConfirmSubmitting} type="button">
                  {isConfirmSubmitting ? "Saving…" : "Confirm & Save"}
                </button>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="hr-input-bar">
        <textarea
          ref={inputRef}
          className="hr-input"
          placeholder={streamState.isStreaming ? "Agent is responding…" : "Message the HR assistant…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={streamState.isStreaming}
          rows={2}
        />
        <button
          className="hr-send-btn"
          onClick={handleSend}
          disabled={streamState.isStreaming || !input.trim()}
          type="button"
        >
          {streamState.isStreaming ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
