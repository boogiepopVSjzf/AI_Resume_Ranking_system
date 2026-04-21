export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export const NAV_ITEMS = [
  {
    id: "home",
    label: "Home",
    description: "Overview",
  },
  {
    id: "resume-upload",
    label: "Resume Upload",
    description: "Upload",
  },
  {
    id: "schema-studio",
    label: "Schema Studio",
    description: "Rules",
  },
  {
    id: "scoring-search",
    label: "Scoring Search",
    description: "Rank",
  },
  {
    id: "system-settings",
    label: "System Settings",
    description: "Status",
  },
] as const;

export type RouteId = (typeof NAV_ITEMS)[number]["id"];
