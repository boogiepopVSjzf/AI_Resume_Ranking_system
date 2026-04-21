import { ChangeEvent, useMemo, useState } from "react";
import { runScoringSearch, submitScoringFeedback } from "./scoringSearchApi";
import type {
  FeedbackLabel,
  ScoringResult,
  ScoringSearchResponse,
} from "./scoringSearchTypes";

const FEEDBACK_LABELS: FeedbackLabel[] = ["excellent", "good", "qualified", "bad"];

function formatPercent(value?: number) {
  if (typeof value !== "number") return "n/a";
  return `${Math.round(value * 100)}%`;
}

function formatScore(value?: number) {
  if (typeof value !== "number") return "n/a";
  return value.toFixed(2).replace(/\.00$/, "");
}

function formatRuntime(seconds: number | null) {
  if (seconds === null) return "n/a";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

function ruleTitle(ruleKey: string) {
  return ruleKey.replace(/^rules/i, "Rule ");
}

function isPdf(file: File) {
  return file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf";
}

export function ScoringSearchPage() {
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [hrNote, setHrNote] = useState("");
  const [initialTopK, setInitialTopK] = useState("5");
  const [feedbackExamplesPerLabel, setFeedbackExamplesPerLabel] = useState("2");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<ScoringSearchResponse | null>(null);
  const [runtimeSeconds, setRuntimeSeconds] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackDrafts, setFeedbackDrafts] = useState<
    Record<string, { label: FeedbackLabel; text: string; status?: string }>
  >({});

  const sortedResults = useMemo(() => {
    return [...(result?.results ?? [])].sort((a, b) => b.score - a.score);
  }, [result]);

  const validationError = useMemo(() => {
    if (!jdFile) return "Please upload a JD PDF.";
    if (!isPdf(jdFile)) return "The JD must be a PDF file.";
    if (!Number.isInteger(Number(initialTopK)) || Number(initialTopK) <= 0) {
      return "Top K must be a positive integer.";
    }
    if (
      !Number.isInteger(Number(feedbackExamplesPerLabel)) ||
      Number(feedbackExamplesPerLabel) < 0
    ) {
      return "Examples per label must be 0 or a positive integer.";
    }
    return null;
  }, [feedbackExamplesPerLabel, initialTopK, jdFile]);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setJdFile(file);
    setResult(null);
    setError(null);
  };

  const submit = async () => {
    if (validationError) {
      setError(validationError);
      return;
    }
    if (!jdFile) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);
    setRuntimeSeconds(null);
    setFeedbackDrafts({});
    const startedAt = performance.now();
    try {
      const response = await runScoringSearch({
        jdFile,
        hrNote,
        initialTopK: Number(initialTopK),
        feedbackExamplesPerLabel: Number(feedbackExamplesPerLabel),
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scoring search failed.");
    } finally {
      setRuntimeSeconds((performance.now() - startedAt) / 1000);
      setIsSubmitting(false);
    }
  };

  const updateFeedback = (
    resumeId: string,
    patch: Partial<{ label: FeedbackLabel; text: string; status: string }>,
  ) => {
    setFeedbackDrafts((current) => ({
      ...current,
      [resumeId]: {
        label: current[resumeId]?.label ?? "qualified",
        text: current[resumeId]?.text ?? "",
        ...patch,
      },
    }));
  };

  const submitFeedback = async (scoreResult: ScoringResult) => {
    if (!result?.schema.schema_id) return;
    const draft = feedbackDrafts[scoreResult.resume_id] ?? {
      label: "qualified" as FeedbackLabel,
      text: "",
    };
    updateFeedback(scoreResult.resume_id, { status: "Saving..." });
    try {
      await submitScoringFeedback({
        schema_id: result.schema.schema_id,
        resume_id: scoreResult.resume_id,
        label: draft.label,
        feedback_text: draft.text,
        score: scoreResult.score,
        scoring_result: scoreResult,
      });
      updateFeedback(scoreResult.resume_id, { status: "Feedback saved" });
    } catch (err) {
      updateFeedback(scoreResult.resume_id, {
        status: err instanceof Error ? err.message : "Feedback failed",
      });
    }
  };

  return (
    <section className="scoring-search page-grid">
      <div className="scoring-hero">
        <p className="eyebrow">Scoring Search</p>
        <h2>Rank candidates.</h2>
      </div>

      <div className="scoring-layout">
        <div className="form-panel scoring-form">
          <label className="field-label" htmlFor="jd-file">
            Job description PDF
          </label>
          <label className="jd-file-picker" htmlFor="jd-file">
            <input id="jd-file" type="file" accept=".pdf,application/pdf" onChange={onFileChange} />
            <span>{jdFile ? jdFile.name : "Choose JD PDF"}</span>
          </label>

          <label className="field-label" htmlFor="hr-note">
            HR note
          </label>
          <textarea
            id="hr-note"
            className="text-area"
            value={hrNote}
            onChange={(event) => setHrNote(event.target.value)}
            placeholder="Optional context, preferences, or must-have requirements..."
            rows={5}
          />

          <div className="compact-fields">
            <label className="inline-control">
              <span>Initial top K</span>
              <input
                className="text-input"
                value={initialTopK}
                onChange={(event) => setInitialTopK(event.target.value)}
                inputMode="numeric"
              />
            </label>
            <label className="inline-control">
              <span>Examples per label</span>
              <input
                className="text-input"
                value={feedbackExamplesPerLabel}
                onChange={(event) => setFeedbackExamplesPerLabel(event.target.value)}
                inputMode="numeric"
              />
            </label>
          </div>

          <button
            type="button"
            className="primary-action"
            onClick={submit}
            disabled={isSubmitting}
          >
            {isSubmitting ? "Scoring candidates..." : "Run scoring search"}
          </button>

          {error && <div className="error-panel">{error}</div>}
        </div>

        <aside className="scoring-status-panel">
          <p className="eyebrow">Pipeline</p>
          <div className={`pipeline-stage ${isSubmitting ? "active" : ""}`}>
            <span>01</span>
            <strong>Rewrite JD context</strong>
          </div>
          <div className={`pipeline-stage ${isSubmitting ? "active" : ""}`}>
            <span>02</span>
            <strong>Retrieve resumes</strong>
          </div>
          <div className={`pipeline-stage ${isSubmitting ? "active" : ""}`}>
            <span>03</span>
            <strong>Match schema</strong>
          </div>
          <div className={`pipeline-stage ${isSubmitting ? "active" : ""}`}>
            <span>04</span>
            <strong>Score</strong>
          </div>
          {result && (
            <div className="schema-match-card">
              <span>Matched schema</span>
              <strong>{result.schema.schema_name}</strong>
              <small>{result.feedback_examples_count} examples</small>
            </div>
          )}
        </aside>
      </div>

      {result && (
        <div className="score-results">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Ranked Output</p>
              <h2>{result.count} candidates scored</h2>
            </div>
            <div className="runtime-pill">
              <span>Total runtime</span>
              <strong>{isSubmitting ? "Timing..." : formatRuntime(runtimeSeconds)}</strong>
            </div>
          </div>

          {sortedResults.length === 0 ? (
            <div className="empty-result">
              <span />
              <p>No candidates found.</p>
            </div>
          ) : (
            sortedResults.map((scoreResult, index) => (
              <article className="candidate-card" key={scoreResult.resume_id}>
                <div className="candidate-rank">
                  <span>#{index + 1}</span>
                  <strong>{formatScore(scoreResult.score)}</strong>
                  <small>/ 10</small>
                </div>
                <div className="candidate-body">
                  <div className="candidate-header">
                    <div>
                      <p className="eyebrow">Resume ID</p>
                      <h3>{scoreResult.resume_id}</h3>
                    </div>
                    <div className="similarity-pill">
                      Similarity {formatPercent(scoreResult.retrieval?.similarity_score)}
                    </div>
                  </div>

                  <div className="score-overview compact">
                    <div>
                      <span>Total score</span>
                      <strong>{formatScore(scoreResult.score)} / 10</strong>
                    </div>
                  </div>

                  {scoreResult.rule_scores && (
                    <div className="rule-breakdown compact">
                      {Object.entries(scoreResult.rule_scores).map(([ruleKey, rule]) => (
                        <article className="rule-breakdown-card" key={ruleKey}>
                          <div className="rule-breakdown-head">
                            <div>
                              <span>{ruleTitle(ruleKey)}</span>
                              <h4>Weight {formatPercent(rule.weight)}</h4>
                            </div>
                            <strong>{formatScore(rule.score)} / 10</strong>
                          </div>
                          <p>{rule.reason}</p>
                        </article>
                      ))}
                    </div>
                  )}

                  <details className="overall-note">
                    <summary>Overall explanation</summary>
                    <p>{scoreResult.explanation}</p>
                  </details>

                  <div className="feedback-box">
                    <div>
                      <p className="eyebrow">Feedback</p>
                      <span>Save judgment.</span>
                    </div>
                    <div className="feedback-controls">
                      <select
                        value={feedbackDrafts[scoreResult.resume_id]?.label ?? "qualified"}
                        onChange={(event) =>
                          updateFeedback(scoreResult.resume_id, {
                            label: event.target.value as FeedbackLabel,
                          })
                        }
                      >
                        {FEEDBACK_LABELS.map((label) => (
                          <option value={label} key={label}>
                            {label}
                          </option>
                        ))}
                      </select>
                      <input
                        value={feedbackDrafts[scoreResult.resume_id]?.text ?? ""}
                        onChange={(event) =>
                          updateFeedback(scoreResult.resume_id, { text: event.target.value })
                        }
                        placeholder="Optional feedback text"
                      />
                      <button type="button" onClick={() => submitFeedback(scoreResult)}>
                        Save
                      </button>
                    </div>
                    {feedbackDrafts[scoreResult.resume_id]?.status && (
                      <small>{feedbackDrafts[scoreResult.resume_id].status}</small>
                    )}
                  </div>
                </div>
              </article>
            ))
          )}
        </div>
      )}
    </section>
  );
}
