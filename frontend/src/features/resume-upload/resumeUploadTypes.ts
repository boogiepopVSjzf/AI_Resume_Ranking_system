export type BatchParseSuccess = {
  filename: string;
  resume_id?: string;
  persisted_to_db?: boolean;
};

export type BatchParseFailure = {
  filename: string;
  reason: string;
};

export type BatchParseResponse = {
  total: number;
  succeeded_count: number;
  failed_count: number;
  succeeded: BatchParseSuccess[];
  failed: BatchParseFailure[];
};
