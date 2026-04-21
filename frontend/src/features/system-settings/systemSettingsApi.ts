import { API_BASE_URL } from "../../lib/constants";
import type { SystemStatusResponse } from "./systemSettingsTypes";

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/api/system/status`);
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }
  return response.json();
}
