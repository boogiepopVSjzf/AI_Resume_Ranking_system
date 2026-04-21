export type RuleScore = {
  score: number;
  weight: number;
  weighted_score: number;
  reason: string;
};

export type ScoringResult = {
  resume_id: string;
  score: number;
  rule_scores?: Record<string, RuleScore>;
  explanation: string;
  retrieval?: {
    similarity_score?: number;
  };
};

export type ScoringSchema = {
  schema_id: string;
  schema_name: string;
  summary?: string;
  rules_json?: Record<
    string,
    {
      rule_text?: string;
      weight?: number;
    }
  >;
};

export type ScoringSearchResponse = {
  schema: ScoringSchema;
  feedback_examples_count: number;
  search_query: string;
  filtered_resume_count?: number;
  retrieved_resume_ids?: string[];
  count: number;
  results: ScoringResult[];
};

export type FeedbackLabel = "excellent" | "good" | "qualified" | "bad";

export type FeedbackPayload = {
  schema_id: string;
  resume_id: string;
  label: FeedbackLabel;
  feedback_text?: string;
  score: number;
  scoring_result: ScoringResult;
};
