import { useEffect, useState } from "react";
import { fetchSystemStatus } from "./systemSettingsApi";
import type { RoleLlmStatus, SystemStatusResponse } from "./systemSettingsTypes";

function StatusPill({ ok, label }: { ok: boolean; label?: string }) {
  return (
    <span className={`settings-pill ${ok ? "ok" : "warn"}`}>
      {label ?? (ok ? "Configured" : "Missing")}
    </span>
  );
}

function BoolLine({ label, value }: { label: string; value: boolean }) {
  return (
    <div className="settings-line">
      <span>{label}</span>
      <StatusPill ok={value} label={value ? "On" : "Off"} />
    </div>
  );
}

function LlmRouteCard({ route }: { route: RoleLlmStatus }) {
  return (
    <article className="llm-route-card">
      <div>
        <p className="eyebrow">{route.label}</p>
        <h3>{route.provider}</h3>
      </div>
      <div className="settings-line">
        <span>Model</span>
        <strong>{route.model}</strong>
      </div>
      <div className="settings-line">
        <span>API key</span>
        <StatusPill ok={route.api_key_configured} />
      </div>
      <div className="settings-line">
        <span>Temperature</span>
        <strong>{route.temperature}</strong>
      </div>
    </article>
  );
}

export function SystemSettingsPage() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadStatus = async () => {
    setIsLoading(true);
    setError(null);
    try {
      setStatus(await fetchSystemStatus());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load system status.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadStatus();
  }, []);

  return (
    <section className="system-settings page-grid">
      <div className="settings-hero">
        <p className="eyebrow">System Settings</p>
        <h2>System status.</h2>
      </div>

      <div className="settings-toolbar">
        <div>
          <p className="eyebrow">Runtime</p>
          <strong>{status?.runtime.mode ?? "read_only"}</strong>
          <span>Restart backend after .env changes.</span>
        </div>
        <button type="button" className="secondary-action" onClick={loadStatus}>
          {isLoading ? "Refreshing..." : "Refresh status"}
        </button>
      </div>

      {error && <div className="error-panel">{error}</div>}

      {status && (
        <>
          {status.warnings.length > 0 && (
            <div className="settings-warning-panel">
              <p className="eyebrow">Warnings</p>
              {status.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          )}

          <div className="settings-grid">
            <article className="settings-card">
              <p className="eyebrow">Database</p>
              <h3>Postgres persistence</h3>
              <BoolLine label="Persistence" value={status.database.enabled} />
              <div className="settings-line">
                <span>DATABASE_URL</span>
                <StatusPill ok={status.database.url_configured} />
              </div>
              <div className="settings-line">
                <span>SSL mode</span>
                <strong>{status.database.sslmode}</strong>
              </div>
              <BoolLine label="Auto init" value={status.database.auto_init} />
            </article>

            <article className="settings-card">
              <p className="eyebrow">Object Storage</p>
              <h3>AWS S3</h3>
              <BoolLine label="S3 storage" value={status.s3.enabled} />
              <div className="settings-line">
                <span>Bucket</span>
                <strong>{status.s3.bucket ?? "Not set"}</strong>
              </div>
              <div className="settings-line">
                <span>Region</span>
                <strong>{status.s3.region ?? "Not set"}</strong>
              </div>
              <div className="settings-line">
                <span>Credentials</span>
                <StatusPill ok={status.s3.credentials_configured} />
              </div>
              <BoolLine label="Custom endpoint" value={status.s3.endpoint_configured} />
            </article>

            <article className="settings-card">
              <p className="eyebrow">Embedding</p>
              <h3>Semantic vectors</h3>
              <div className="settings-line">
                <span>Model</span>
                <strong>{status.embedding.model}</strong>
              </div>
              <div className="settings-line">
                <span>Device</span>
                <strong>{status.embedding.device}</strong>
              </div>
              <div className="settings-line">
                <span>Dimension</span>
                <strong>{status.embedding.dimension}</strong>
              </div>
              <BoolLine label="Preload" value={status.embedding.preload} />
            </article>

            <article className="settings-card">
              <p className="eyebrow">Reranker</p>
              <h3>Precision ranking</h3>
              <BoolLine label="Enabled" value={status.reranker.enabled} />
              <div className="settings-line">
                <span>Model</span>
                <strong>{status.reranker.model}</strong>
              </div>
              <div className="settings-line">
                <span>Device</span>
                <strong>{status.reranker.device}</strong>
              </div>
              <div className="settings-line">
                <span>Candidate pool</span>
                <strong>{status.reranker.candidate_pool_size}</strong>
              </div>
              <BoolLine label="Preload" value={status.reranker.preload} />
            </article>

            <article className="settings-card">
              <p className="eyebrow">Upload Limits</p>
              <h3>Resume intake</h3>
              <div className="settings-line">
                <span>Max upload</span>
                <strong>{status.limits.max_upload_mb} MB</strong>
              </div>
              <div className="settings-line">
                <span>Max batch</span>
                <strong>{status.limits.max_batch_size}</strong>
              </div>
              <div className="settings-line">
                <span>Allowed files</span>
                <strong>{status.limits.allowed_extensions.join(", ")}</strong>
              </div>
            </article>
          </div>

          <div className="llm-routing-panel">
            <div>
              <p className="eyebrow">Model Routing</p>
              <h2>LLM routing</h2>
            </div>
            <div className="llm-route-grid">
              <LlmRouteCard route={status.llm_routing.parse_query} />
              <LlmRouteCard route={status.llm_routing.schema} />
              <LlmRouteCard route={status.llm_routing.scoring} />
            </div>
          </div>
        </>
      )}

      {isLoading && !status && (
        <div className="empty-result">
          <span />
          <p>Loading runtime status from the backend...</p>
        </div>
      )}
    </section>
  );
}
