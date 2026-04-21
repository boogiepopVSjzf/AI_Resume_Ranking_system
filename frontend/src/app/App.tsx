import { useMemo, useState } from "react";
import { AppShell } from "./AppShell";
import type { RouteId } from "../lib/constants";
import { HomePage } from "../features/home/HomePage";
import { PlaceholderPage } from "../components/common/PlaceholderPage";
import { SchemaStudioPage } from "../features/schema-studio/SchemaStudioPage";
import { ResumeUploadPage } from "../features/resume-upload/ResumeUploadPage";
import { ScoringSearchPage } from "../features/scoring-search/ScoringSearchPage";
import { SystemSettingsPage } from "../features/system-settings/SystemSettingsPage";

export function App() {
  const [activeRoute, setActiveRoute] = useState<RouteId>("home");

  const page = useMemo(() => {
    switch (activeRoute) {
      case "home":
        return <HomePage onNavigate={setActiveRoute} />;
      case "resume-upload":
        return <ResumeUploadPage />;
      case "schema-studio":
        return <SchemaStudioPage />;
      case "scoring-search":
        return <ScoringSearchPage />;
      case "system-settings":
        return <SystemSettingsPage />;
      default:
        return null;
    }
  }, [activeRoute]);

  return (
    <AppShell activeRoute={activeRoute} onNavigate={setActiveRoute}>
      {page}
    </AppShell>
  );
}
