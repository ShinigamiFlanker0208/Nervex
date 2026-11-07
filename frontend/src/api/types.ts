// Mirrors api/schemas.py. Keep in sync by hand; /docs on the API is the source of truth.
export interface Paper { paper_id: string; title: string; nodes: number; edges: number }
export interface GNode { id: string; name: string | null; type: string | null; description: string | null; degree: number }
export interface GEdge { source: string; target: string; type: string }
export interface GraphOverview { paper_id: string; nodes: GNode[]; edges: GEdge[] }
export interface Component { id: string; name: string | null; description: string | null; degree: number; formulas: number }

export interface IngestJob {
  job_id: string; filename: string; state: "queued" | "running" | "done" | "failed";
  progress: string[]; error: string | null;
  result: { paper_id: string; paper_title: string; num_pages: number; num_entities_extracted: number; num_relationships_merged: number; status_message: string } | null;
}

export type NodeName = "start" | "researcher" | "architect" | "coder" | "sandbox_executor" | "critic" | "end";
export interface RunEvent {
  i: number; t: number; node: NodeName;
  node_ids?: string[]; formulas?: number; hyperparameters?: string[]; formulas_tex?: { id: string; latex: string }[];
  expected_inputs?: Record<string, string>; skeleton?: string;
  code?: string; lines?: number; attempt?: number | null;
  status?: string; layers?: string[]; log_tail?: string; feedback?: string;
}
export interface LayerTrace { input_shape?: number[]; output_shape?: number[]; [k: string]: unknown }
export interface GenResult {
  paper_id: string; target_component: string; target_framework: string; generated_code: string; attempt_count: number;
  execution_status: string; tensor_shape_traces: Record<string, any>; error_logs: any[]; error_type: string | null; error_details: string | null;
}
export interface RunStatus {
  run_id: string; state: "queued" | "running" | "done" | "failed"; meta: Record<string, string>;
  events: RunEvent[]; total_events: number; started_at: number | null; created_at: number; result: GenResult | null;
}
export interface ExplainResult { answer_markdown: string; cited_graph_nodes: string[]; cited_paper_passages: string[]; inferred_tensor_shapes: Record<string, string> | null }
export interface AblationResult {
  mutation_prompt: string; baseline_code: string; ablated_code: string;
  baseline_metrics: Record<string, any>; ablated_metrics: Record<string, any>; diff_explanation: string;
}
export interface TranspileResult {
  source_framework: string; target_framework: string; transpiled_code: string; syntax_valid: boolean; notes: string;
  runtime_verified: boolean | null; runtime_logs: string; output_shape: unknown;
}
export interface Health {
  status: string; message: string; subsystems: Record<string, boolean>;
  llm: { provider?: string; model?: string; has_key?: boolean; error?: string };
}
export interface RunSummary { run_id: string; paper_id: string; component: string; framework: string; status: string; attempts: number; duration: number; layers: number; lines: number; created_at: number }
export interface SystemConfig {
  llm: { provider?: string; model?: string };
  sandbox: { network: string; capabilities: string; user: string; workspace: string; tmpfs_mb: number; memory_mb: number; timeout_s: number; image: string };
  loop: { max_attempts: number; loop_timeout_s: number };
  ingest: { extraction_concurrency: number; max_failed_fraction: number; max_upload_mb: number; ocr_gpu_fraction: number };
}
export interface Passage { id: string; text: string; header: string }
export interface SavedAblation extends AblationResult { ablation_id: string; paper_id: string; component: string; created_at: number }
