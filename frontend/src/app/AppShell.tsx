import type { ReactNode } from "react";

import { Sidebar } from "../components/layout/Sidebar";
import type { RouteId } from "../lib/constants";

type AppShellProps = {
  activeRoute: RouteId;
  onNavigate: (route: RouteId) => void;
  children: ReactNode;
};

export function AppShell({ activeRoute, onNavigate, children }: AppShellProps) {
  return (
    <div className="app-shell" id="top">
      <div className="orb orb-one" />
      <div className="orb orb-two" />
      <Sidebar activeRoute={activeRoute} onNavigate={onNavigate} />
      <main className="main-canvas">
        <div className="main-canvas-inner">
          <header className="topbar">
            <div>
              <p className="topbar-kicker">AI Resume Ranking System</p>
              <h1>Recruiting Intelligence Console</h1>
            </div>
            <div className="demo-pill">
              <span className="pulse-dot" />
              Local demo mode
            </div>
          </header>
          {children}
        </div>
        <footer className="app-footer" aria-label="Attribution">
          <span>Powered by </span>
          <span className="footer-brand">Entagile</span>
        </footer>
      </main>
    </div>
  );
}
