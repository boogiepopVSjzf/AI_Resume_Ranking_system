export interface RagSearchResponse {
  hard_filters: Record<string, unknown>
  search_query: string
  search_query_embedding: number[]
  filtered_resume_ids: string[]
  top_k: number
  top_k_resume_ids: string[]
  count: number
  query_usage: Record<string, unknown>
}

export interface ParseBatchResponse {
  total: number
  succeeded_count: number
  failed_count: number
  succeeded: unknown[]
  failed: unknown[]
}