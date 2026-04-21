import { API_BASE_URL } from "./constants";

export type HealthResponse = {
  message: string;
  docs?: string;
  endpoints?: string[];
};

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  return response.json();
}
