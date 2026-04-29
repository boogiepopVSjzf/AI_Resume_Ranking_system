import { useEffect, useMemo, useState } from "react";
import { fetchHealth, type HealthResponse } from "../../lib/api";
import { API_BASE_URL, NAV_ITEMS, type RouteId } from "../../lib/constants";

type HomePageProps = {
  onNavigate: (route: RouteId) => void;
};

type LoadState =
  | { status: "loading" }
  | { status: "ready"; data: HealthResponse }
  | { status: "error"; error: string };

const flowSteps = [
  "Resume parsing",
  "Structured storage",
  "RAG retrieval",
  "Schema scoring",
  "Human feedback",
];

export function HomePage({ onNavigate }: HomePageProps) {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    let mounted = true;
    fetchHealth()
      .then((data) => {
        if (mounted) setState({ status: "ready", data });
      })
      .catch((error: unknown) => {
        if (mounted) {
          setState({
            status: "error",
            error: error instanceof Error ? error.message : "Unknown backend error",
          });
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  const endpointCount = useMemo(() => {
    return state.status === "ready" ? state.data.endpoints?.length ?? 0 : 0;
  }, [state]);

  return (
    <section className="home-page">
      <div className="hero-grid">
        <div className="hero-card">
          <p className="eyebrow">Dashboard</p>
          <h2>Resume ranking, end to end.</h2>
          <div className="hero-actions">
            <button
              className="primary-action"
              type="button"
              onClick={() => onNavigate("schema-studio")}
            >
              Schema Studio
            </button>
            <button
              className="secondary-action"
              type="button"
              onClick={() => onNavigate("scoring-search")}
            >
              Scoring Search
            </button>
          </div>
        </div>

        <div className="signal-card">
          <div className="signal-ring">
            <span />
            <span />
            <span />
            <strong>{state.status === "ready" ? "Ready" : state.status === "loading" ? "…" : "!"}</strong>
          </div>
          <div>
            <p className="eyebrow">Backend Signal</p>
            <h3>
              {state.status === "ready"
                ? "FastAPI is connected"
                : state.status === "loading"
                  ? "Checking backend"
                  : "Backend needs attention"}
            </h3>
            <p className="muted">
              {state.status === "ready"
                ? `${endpointCount} endpoints online`
                : state.status === "loading"
                  ? "Checking API..."
                  : state.error}
            </p>
          </div>
        </div>
      </div>

      <div className="status-strip">
        <div>
          <span>API Base URL</span>
          <strong>{API_BASE_URL}</strong>
        </div>
        <div>
          <span>Docs</span>
          <strong>{state.status === "ready" ? state.data.docs ?? "/docs" : "/docs"}</strong>
        </div>
        <div>
          <span>Mode</span>
          <strong>Local first</strong>
        </div>
      </div>

      <div className="section-heading">
        <div>
          <p className="eyebrow">Workspace Map</p>
          <h2>Modules</h2>
        </div>
      </div>

      <div className="launch-grid">
        {NAV_ITEMS.filter((item) => item.id !== "home").map((item) => (
          <button
            className="launch-card"
            key={item.id}
            onClick={() => onNavigate(item.id)}
            type="button"
          >
            <span>{item.label}</span>
            <strong>{item.description}</strong>
          </button>
        ))}
      </div>

      <div className="flow-panel">
        <p className="eyebrow">Flow</p>
        <div className="flow-line">
          {flowSteps.map((step, index) => (
            <div className="flow-node" key={step}>
              <span>{index + 1}</span>
              <strong>{step}</strong>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
