import type * as T from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number, public kind?: string) { super(message); }
}

async function req<R>(path: string, init?: RequestInit): Promise<R> {
  let res: Response;
  try { res = await fetch(path, init); }
  catch { throw new ApiError("Cannot reach the Nervex API. Is `uv run python main.py` running?", 0, "offline"); }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`, kind: string | undefined;
    try {
      const b = await res.json();
      kind = b.error ?? b.kind;
      msg = typeof b.detail === "string" ? b.detail : b.message ?? b.detail?.[0]?.msg ?? msg;
    } catch { /* non-JSON body */ }
    throw new ApiError(msg, res.status, kind);
  }
  return res.json() as Promise<R>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
const V = "/api/v1";

export const api = {
  health: () => req<T.Health>("/_health"),
  papers: () => req<T.Paper[]>(`${V}/papers`),
  graph: (pid: string, limit = 140, include: string[] = []) =>
    req<T.GraphOverview>(`${V}/papers/${pid}/graph?limit=${limit}${include.length ? `&include=${encodeURIComponent(include.join(","))}` : ""}`),
  components: (pid: string) => req<T.Component[]>(`${V}/papers/${pid}/components`),
  neighborhood: (pid: string, node_id: string, depth = 1) =>
    req<{ nodes: any[]; edges: any[] }>(`${V}/graph/query`, json({ paper_id: pid, node_id, depth })),
  ingest: (file: File) => {
    const fd = new FormData(); fd.append("file", file);
    return req<{ job_id: string }>(`${V}/ingest/pdf`, { method: "POST", body: fd });
  },
  ingestStatus: (id: string) => req<T.IngestJob>(`${V}/ingest/status/${id}`),
  startRun: (paper_id: string, target_component: string, target_framework = "PyTorch") =>
    req<{ run_id: string }>(`${V}/generate/runs`, json({ paper_id, target_component, target_framework })),
  history: (pid: string) => req<T.RunSummary[]>(`${V}/generate/history?paper_id=${pid}`),
  historyItem: (id: string) => req<T.RunStatus>(`${V}/generate/history/${id}`),
  passages: (pid: string) => req<T.Passage[]>(`${V}/papers/${pid}/passages`),
  ablations: (pid: string) => req<T.SavedAblation[]>(`${V}/features/ablations?paper_id=${pid}`),
  config: () => req<T.SystemConfig>(`${V}/system/config`),
  run: (id: string, since: number) => req<T.RunStatus>(`${V}/generate/runs/${id}?since=${since}`),
  explain: (b: { paper_id: string; user_question: string; start: number; end: number; code: string; history: { role: string; content: string }[] }) =>
    req<T.ExplainResult>(`${V}/explainer/chat`, json({
      paper_id: b.paper_id, user_question: b.user_question, highlighted_line_start: b.start,
      highlighted_line_end: b.end, current_code_script: b.code, chat_history: b.history,
    })),
  ablation: (b: { paper_id: string; target_component: string; baseline_code: string; expected_inputs: Record<string, string>; mutation_prompt: string }) =>
    req<T.AblationResult>(`${V}/features/ablation`, json(b)),
  transpile: (b: { source_code: string; source_framework: string; target_framework: string; architecture_skeleton: string; expected_inputs?: Record<string, string> }) =>
    req<T.TranspileResult>(`${V}/features/transpile`, json(b)),
};
