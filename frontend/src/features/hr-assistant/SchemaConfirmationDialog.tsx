import { useState } from "react";
import type { SchemaDraft } from "./hrAssistantTypes";

type SchemaConfirmationDialogProps = {
  draft: SchemaDraft;
  isSubmitting: boolean;
  onConfirm: () => void;
  onReject: () => void;
};

export function SchemaConfirmationDialog({
  draft,
  isSubmitting,
  onConfirm,
  onReject,
}: SchemaConfirmationDialogProps) {
  const [expanded, setExpanded] = useState(false);
  const rules = Object.entries(draft.rules_json ?? {});

  return (
    <div className="hr-confirm-overlay" role="dialog" aria-modal="true" aria-label="Confirm schema">
      <div className="hr-confirm-modal">
        <div className="hr-confirm-header">
          <p className="eyebrow">Schema Draft</p>
          <h3 className="hr-confirm-title">{draft.schema_name}</h3>
        </div>

        <p className="hr-confirm-summary">{draft.summary}</p>

        <div className="hr-confirm-rules">
          {rules.map(([key, rule]) => (
            <div key={key} className="hr-confirm-rule">
              <div className="hr-confirm-rule-header">
                <span className="hr-confirm-rule-key">{key}</span>
                <span className="hr-confirm-rule-weight">{Math.round((rule.weight ?? 0) * 100)}%</span>
              </div>
              <p className="hr-confirm-rule-text">{rule.rule_text}</p>
            </div>
          ))}
        </div>

        {rules.length > 3 && (
          <button
            className="hr-confirm-expand"
            onClick={() => setExpanded((v) => !v)}
            type="button"
          >
            {expanded ? "Show less" : `Show all ${rules.length} rules`}
          </button>
        )}

        <div className="hr-confirm-actions">
          <button
            className="btn-secondary"
            onClick={onReject}
            disabled={isSubmitting}
            type="button"
          >
            Reject
          </button>
          <button
            className="btn-primary"
            onClick={onConfirm}
            disabled={isSubmitting}
            type="button"
          >
            {isSubmitting ? "Saving…" : "Confirm & Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
