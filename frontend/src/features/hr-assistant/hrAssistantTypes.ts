export type AgentEvent =
  | { event: "session_resolved"; data: { session_id: string; resumed: boolean } }
  | { event: "thinking"; data: string }
  | { event: "tool_start"; data: { tool: string; args: Record<string, unknown> } }
  | { event: "tool_result"; data: { tool: string; output: unknown } }
  | { event: "confirmation_required"; data: SchemaDraft }
  | { event: "text_chunk"; data: string }
  | { event: "done"; data: { session_id: string; message_id: string } }
  | { event: "error"; data: { message: string } };

export type ChatMessage = {
  message_id: string;
  session_id: string;
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  tool_name?: string | null;
  tool_call_id?: string | null;
  tool_calls_json?: unknown[] | null;
  created_at: string | null;
};

export type ChatSession = {
  session_id: string;
  title: string | null;
  active_schema_id: string | null;
  hr_auto_mode: boolean;
  provider: string | null;
  model: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type SessionDetail = ChatSession & {
  messages: ChatMessage[];
};

export type SchemaDraft = {
  schema_name: string;
  rules_text: string;
  rules_json: Record<string, { rule_text: string; weight: number; raw_text: string }>;
  summary: string;
};

export type SchemaInfo = {
  schema_id: string;
  schema_name: string;
  summary: string;
  version: number;
  is_active: boolean;
};

// Streaming render state (built up in the UI as SSE events arrive)
export type StreamState = {
  isStreaming: boolean;
  pendingText: string;
  thinkingLabel: string;
  activeToolName: string | null;
  pendingConfirmation: SchemaDraft | null;
};
