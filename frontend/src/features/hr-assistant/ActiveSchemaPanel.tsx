import type { SchemaInfo } from "./hrAssistantTypes";

type ActiveSchemaPanelProps = {
  schema: SchemaInfo | null;
  isLoading: boolean;
};

export function ActiveSchemaPanel({ schema, isLoading }: ActiveSchemaPanelProps) {
  return (
    <div className="hr-schema-panel">
      <p className="eyebrow" style={{ marginBottom: "0.6rem" }}>Active Schema</p>
      {isLoading && <p className="muted" style={{ fontSize: "0.8rem" }}>Loading…</p>}
      {!isLoading && !schema && (
        <div className="hr-schema-empty">
          <p>No schema active</p>
          <small>Ask the assistant to create one</small>
        </div>
      )}
      {!isLoading && schema && (
        <div className="hr-schema-info">
          <div className="hr-schema-name">{schema.schema_name}</div>
          <div className="hr-schema-version">v{schema.version}</div>
          <p className="hr-schema-summary">{schema.summary}</p>
        </div>
      )}
    </div>
  );
}
