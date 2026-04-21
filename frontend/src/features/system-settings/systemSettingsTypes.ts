export type RoleLlmStatus = {
  label: string;
  provider: string;
  model: string;
  api_key_configured: boolean;
  temperature: number;
};

export type SystemStatusResponse = {
  runtime: {
    mode: string;
    env_loaded_from: string;
    restart_required_after_env_change: boolean;
  };
  database: {
    enabled: boolean;
    url_configured: boolean;
    sslmode: string;
    auto_init: boolean;
  };
  s3: {
    enabled: boolean;
    bucket_configured: boolean;
    bucket: string | null;
    region_configured: boolean;
    region: string | null;
    endpoint_configured: boolean;
    credentials_configured: boolean;
  };
  llm_routing: {
    parse_query: RoleLlmStatus;
    schema: RoleLlmStatus;
    scoring: RoleLlmStatus;
  };
  embedding: {
    model: string;
    device: string;
    dimension: number;
    preload: boolean;
    include_embedding_in_response: boolean;
  };
  limits: {
    allowed_extensions: string[];
    max_upload_mb: number;
    max_batch_size: number;
  };
  warnings: string[];
};
