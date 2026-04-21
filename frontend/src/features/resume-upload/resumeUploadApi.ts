import { API_BASE_URL } from "../../lib/constants";
import type { BatchParseResponse } from "./resumeUploadTypes";

export async function uploadResumeBatch(files: File[]): Promise<BatchParseResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE_URL}/api/parse/batch`, {
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
