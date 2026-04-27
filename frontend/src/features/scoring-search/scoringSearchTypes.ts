export type RuleScore = {
  score: number;
  weight: number;
  weighted_score: number;
  reason: string;
  feedback_used?: boolean;
  feedback_influence?: string;
};

export type FeedbackInfluenceMode = "off" | "on";
export type FilterMode = "strict" | "balanced" | "semantic_only";

export type FeedbackUsageSummary = {
  mode: FeedbackInfluenceMode;
  used_feedback_ids: string[];
  ignored_feedback_ids: string[];
  overall_influence: string;
};

export type ScoringResult = {
  resume_id: string;
  candidate_name?: string;
  score: number;
  rule_scores?: Record<string, RuleScore>;
  feedback_usage_summary?: FeedbackUsageSummary;
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
  feedback_influence_mode: FeedbackInfluenceMode;
  filter_mode: FilterMode;
  applied_hard_filters?: {
    min_yoe?: number | null;
    required_skills?: string[];
    education_level?: string | null;
    major?: string | null;
  };
  search_query: string;
  filtered_resume_count?: number;
  retrieved_resume_ids?: string[];
  count: number;
  results: ScoringResult[];
};

export type FeedbackLabel = "excellent" | "good" | "qualified" | "bad" | "n/a";

export type FeedbackPayload = {
  schema_id: string;
  resume_id: string;
  label: FeedbackLabel;
  feedback_text?: string;
  score: number;
  scoring_result: ScoringResult;
};
