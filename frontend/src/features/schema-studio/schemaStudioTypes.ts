export type RuleDraft = {
  id: string;
  text: string;
  weight: string;
};

export type ScoringSchemaResponse = {
  schema_id: string;
  schema_name: string;
  rules_json: Record<
    string,
    {
      rule_text: string;
      weight: number;
      raw_text: string;
    }
  >;
  summary: string;
  version: number;
  is_active: boolean;
  summary_embedding_generated: boolean;
  usage?: Record<string, unknown>;
};
