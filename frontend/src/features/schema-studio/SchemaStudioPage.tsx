import { useMemo, useState } from "react";
import { createScoringSchema } from "./schemaStudioApi";
import type { RuleDraft, ScoringSchemaResponse } from "./schemaStudioTypes";

const WEIGHT_TOTAL_TARGET = 100;
const WEIGHT_TOTAL_TOLERANCE = 0.01;

function makeId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function newRule(index: number): RuleDraft {
  return {
    id: makeId(),
    text: "",
    weight: index === 0 ? "25" : "",
  };
}

function buildRulesText(rules: RuleDraft[]) {
  return rules
    .map((rule, index) => {
      return `rules${index + 1}: ${rule.text.trim()} weight: ${rule.weight.trim()}%`;
    })
    .join("\n\n");
}

function parseWeight(weight: string) {
  const value = Number(weight);
  return Number.isFinite(value) ? value : 0;
}

export function SchemaStudioPage() {
  const [schemaName, setSchemaName] = useState("Data Science Level 1");
  const [rules, setRules] = useState<RuleDraft[]>([
    {
      id: makeId(),
      text: "Evaluate SQL querying ability, including joins, aggregations, window functions, and data validation for analytical tasks.",
      weight: "25",
    },
    {
      id: makeId(),
      text: "Evaluate Python data science execution, including data cleaning, exploratory analysis, visualization, and reproducible workflows.",
      weight: "25",
    },
  ]);
  const [result, setResult] = useState<ScoringSchemaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const weightTotal = useMemo(() => {
    return rules.reduce((sum, rule) => sum + parseWeight(rule.weight), 0);
  }, [rules]);

  const rulesText = useMemo(() => buildRulesText(rules), [rules]);

  const validationError = useMemo(() => {
    if (!schemaName.trim()) return "Schema name is required.";
    if (rules.length === 0) return "At least one rule is required.";
    for (const [index, rule] of rules.entries()) {
      if (!rule.text.trim()) return `Rule ${index + 1} needs text.`;
      const weight = Number(rule.weight);
      if (!Number.isFinite(weight) || weight <= 0) {
        return `Rule ${index + 1} needs a positive weight.`;
      }
    }
    if (Math.abs(weightTotal - WEIGHT_TOTAL_TARGET) > WEIGHT_TOTAL_TOLERANCE) {
      return `Rule weights must add up to 100%. Current total is ${weightTotal}%.`;
    }
    return null;
  }, [rules, schemaName, weightTotal]);

  const updateRule = (id: string, patch: Partial<RuleDraft>) => {
    setRules((current) =>
      current.map((rule) => (rule.id === id ? { ...rule, ...patch } : rule)),
    );
  };

  const removeRule = (id: string) => {
    setRules((current) => current.filter((rule) => rule.id !== id));
  };

  const addRule = () => {
    setRules((current) => [...current, newRule(current.length)]);
  };

  const submit = async () => {
    if (validationError) {
      setError(validationError);
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      const created = await createScoringSchema({
        schemaName: schemaName.trim(),
        rulesText,
      });
      setResult(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create schema");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="schema-studio page-grid">
      <div className="studio-intro">
        <p className="eyebrow">Schema Studio</p>
        <h2>Create scoring rules.</h2>
      </div>

      <div className="schema-layout">
        <div className="form-panel">
          <label className="field-label" htmlFor="schema-name">
            Schema name
          </label>
          <input
            id="schema-name"
            className="text-input"
            value={schemaName}
            onChange={(event) => setSchemaName(event.target.value)}
            placeholder="e.g. Data Science Level 1"
          />

          <div className="rule-toolbar">
            <div>
              <p className="eyebrow">Rules</p>
              <strong>{rules.length} criteria</strong>
            </div>
            <div className={`weight-meter ${Math.round(weightTotal) === 100 ? "balanced" : ""}`}>
              Total weight: {weightTotal || 0}%
            </div>
          </div>
          {Math.abs(weightTotal - WEIGHT_TOTAL_TARGET) > WEIGHT_TOTAL_TOLERANCE && (
            <div className="weight-warning">
              {weightTotal < WEIGHT_TOTAL_TARGET
                ? `Need +${Number((WEIGHT_TOTAL_TARGET - weightTotal).toFixed(2))}% to reach 100%.`
                : `Reduce ${Number((weightTotal - WEIGHT_TOTAL_TARGET).toFixed(2))}% to reach 100%.`}
            </div>
          )}

          <div className="rules-stack">
            {rules.map((rule, index) => (
              <article className="rule-card" key={rule.id}>
                <div className="rule-card-header">
                  <span>rules{index + 1}</span>
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => removeRule(rule.id)}
                    disabled={rules.length === 1}
                  >
                    Remove
                  </button>
                </div>
                <textarea
                  className="text-area"
                  value={rule.text}
                  onChange={(event) => updateRule(rule.id, { text: event.target.value })}
                  placeholder="Describe this scoring criterion..."
                  rows={4}
                />
                <label className="inline-field">
                  <span>Weight</span>
                  <input
                    className="weight-input"
                    value={rule.weight}
                    onChange={(event) => updateRule(rule.id, { weight: event.target.value })}
                    inputMode="decimal"
                    placeholder="25"
                  />
                  <span>%</span>
                </label>
              </article>
            ))}
          </div>

          <div className="form-actions">
            <button type="button" className="secondary-action" onClick={addRule}>
              + Add rule
            </button>
            <button
              type="button"
              className="primary-action"
              onClick={submit}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Creating..." : "Create schema"}
            </button>
          </div>

          {error && <div className="error-panel">{error}</div>}
        </div>

        <aside className="preview-panel">
          <div>
            <p className="eyebrow">Preview</p>
            <h3>{schemaName || "Untitled schema"}</h3>
          </div>
          <div className="payload-preview compact-preview">
            <span>Rules</span>
            <strong>{rules.length}</strong>
            <span>Total weight</span>
            <strong>{weightTotal || 0}%</strong>
          </div>

          {result ? (
            <div className="result-panel">
              <p className="eyebrow">Created Schema</p>
              <h3>{result.schema_name}</h3>
              <dl className="result-grid">
                <div>
                  <dt>schema_id</dt>
                  <dd>{result.schema_id}</dd>
                </div>
                <div>
                  <dt>version</dt>
                  <dd>{result.version}</dd>
                </div>
                <div>
                  <dt>active</dt>
                  <dd>{String(result.is_active)}</dd>
                </div>
                <div>
                  <dt>embedding</dt>
                  <dd>{result.summary_embedding_generated ? "generated" : "missing"}</dd>
                </div>
              </dl>
              <div className="summary-box">
                <span>summary</span>
                <p>{result.summary}</p>
              </div>
              <details className="json-details">
                <summary>Details</summary>
                <pre>{JSON.stringify(result.rules_json, null, 2)}</pre>
              </details>
            </div>
          ) : (
            <div className="empty-result">
              <span />
              <p>Created schema appears here.</p>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
