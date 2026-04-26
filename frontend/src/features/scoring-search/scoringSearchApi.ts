import { API_BASE_URL } from "../../lib/constants";
import type {
  FeedbackInfluenceMode,
  FeedbackPayload,
  ScoringSearchResponse,
} from "./scoringSearchTypes";

export async function runScoringSearch(input: {
  jdFile: File;
  hrNote: string;
  initialTopK: number;
  feedbackExamplesPerLabel: number;
  feedbackInfluenceMode: FeedbackInfluenceMode;
}): Promise<ScoringSearchResponse> {
  const formData = new FormData();
  formData.append("jd_file", input.jdFile);
  formData.append("hr_note", input.hrNote);
  formData.append("initial_top_k", String(input.initialTopK));
  formData.append("feedback_examples_per_label", String(input.feedbackExamplesPerLabel));
  formData.append("feedback_influence_mode", input.feedbackInfluenceMode);

  const response = await fetch(`${API_BASE_URL}/api/scoring-search`, {
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

export async function submitScoringFeedback(payload: FeedbackPayload) {
  const response = await fetch(`${API_BASE_URL}/api/scoring-feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail ?? `Backend returned ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return response.json();
}
