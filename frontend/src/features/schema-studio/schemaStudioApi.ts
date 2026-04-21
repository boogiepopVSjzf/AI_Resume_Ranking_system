import { API_BASE_URL } from "../../lib/constants";
import type { ScoringSchemaResponse } from "./schemaStudioTypes";

export async function createScoringSchema(input: {
  schemaName: string;
  rulesText: string;
}): Promise<ScoringSchemaResponse> {
  const formData = new FormData();
  formData.append("schema_name", input.schemaName);
  formData.append("rules", input.rulesText);

  const response = await fetch(`${API_BASE_URL}/api/scoring-schema`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail ?? `Backend returned ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return response.json();
}
