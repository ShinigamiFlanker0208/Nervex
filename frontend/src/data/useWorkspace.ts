// One brain, three faces: every shell renders THIS state. No shell talks to the API on its own,
// so a design can change what things look like but never what they do.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Passage, SavedAblation, RunSummary, AblationResult, Component, ExplainResult, GenResult, GraphOverview, Health, IngestJob, Paper, RunEvent, TranspileResult } from "../api/types";
import { parseTags } from "../lib/codeTags";

export interface ChatTurn { role: "user" | "assistant"; content: string; cites?: string[]; passages?: string[]; error?: boolean }
export type RunPhase = "idle" | "queued" | "running" | "done" | "failed";

const errMsg = (e: unknown) => (e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));

export function useWorkspace() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [papersLoaded, setPapersLoaded] = useState(false);
  const [paperId, setPaperId] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphOverview | null>(null);
  const [components, setComponents] = useState<Component[]>([]);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);     // node the user clicked
  const [hover, setHover] = useState<string | null>(null);           // node the user hovers (graph OR code tag)

  // ---- run ----
  const [phase, setPhase] = useState<RunPhase>("idle");
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [result, setResult] = useState<GenResult | null>(null);
  const [component, setComponent] = useState<Component | null>(null);
  const [framework, setFramework] = useState("PyTorch");
  const [elapsed, setElapsed] = useState(0);
  const [runErr, setRunErr] = useState<string | null>(null);
  const runRef = useRef<{ id: string; since: number; stop: boolean } | null>(null);
  const t0 = useRef(0);

  // ---- ingest ----
  const [job, setJob] = useState<IngestJob | null>(null);
  const [ingestErr, setIngestErr] = useState<string | null>(null);

  // ---- shelf (saved runs) ----
  const [shelf, setShelf] = useState<RunSummary[]>([]);
  const refreshShelf = useCallback(() => { if (paperId) api.history(paperId).then(setShelf).catch(() => setShelf([])); }, [paperId]);
  useEffect(() => { refreshShelf(); }, [refreshShelf]);

  // ---- paper text + saved experiments ----
  const [passages, setPassages] = useState<Passage[] | null>(null);
  const [ablations, setAblations] = useState<SavedAblation[]>([]);
  const refreshAblations = useCallback(() => { if (paperId) api.ablations(paperId).then(setAblations).catch(() => setAblations([])); }, [paperId]);
  useEffect(() => { setPassages(null); if (paperId) api.passages(paperId).then(setPassages).catch(() => setPassages([])); refreshAblations(); }, [paperId, refreshAblations]);

  // ---- chat / tools ----
  const [chat, setChat] = useState<ChatTurn[]>([]);
  const [asking, setAsking] = useState(false);
  const [ablation, setAblation] = useState<{ busy: boolean; data?: AblationResult; err?: string }>({ busy: false });
  const [transpile, setTranspile] = useState<{ busy: boolean; data?: TranspileResult; err?: string }>({ busy: false });

  const refreshPapers = useCallback(async () => {
    try {
      const p = await api.papers(); setPapers(p); setPapersLoaded(true); setOffline(false); setLoadErr(null);
      setPaperId(cur => cur && p.some(x => x.paper_id === cur) ? cur : p[0]?.paper_id ?? null);
      return p;
    } catch (e) {
      if (e instanceof ApiError && e.kind === "offline") setOffline(true);
      setLoadErr(errMsg(e)); return [];
    }
  }, []);

  useEffect(() => { api.health().then(setHealth).catch(() => setOffline(true)); refreshPapers(); }, [refreshPapers]);

  useEffect(() => {
    setGraph(null); setComponents([]); setSelected(null);
    if (!paperId) return;
    let live = true;
    api.graph(paperId).then(g => live && setGraph(g)).catch(e => live && setLoadErr(errMsg(e)));
    api.components(paperId).then(c => live && setComponents(c)).catch(() => {});
    return () => { live = false; };
  }, [paperId]);

  // A run cites formula nodes that rank below the overview's top-N. Fetch them (only the missing ones, remembered
  // across fetches so later events don't drop earlier extras) so they exist in the graph and can light up.
  const extras = useRef<Set<string>>(new Set());
  const codeNow = useMemo(() => result?.generated_code || [...events].reverse().find(e => e.code)?.code || "", [events, result]);
  const tagsNow = useMemo(() => parseTags(codeNow), [codeNow]);
  useEffect(() => { extras.current = new Set(); }, [paperId]);
  useEffect(() => {
    if (!paperId || !graph || graph.paper_id !== paperId) return;
    const have = new Set(graph.nodes.map(n => n.id));
    const cited = [...new Set([...events.flatMap(e => e.node_ids ?? []), ...tagsNow.map(t => t.nodeId)])];
    const missing = cited.filter(i => !have.has(i));
    if (!missing.length) return;
    missing.forEach(i => extras.current.add(i));
    let live = true;
    api.graph(paperId, 140, [...extras.current]).then(g => live && setGraph(g)).catch(() => {});
    return () => { live = false; };
  }, [events, tagsNow, graph, paperId]);

  // ---------- run ----------
  const stopRun = () => { if (runRef.current) runRef.current.stop = true; };
  useEffect(() => () => stopRun(), []);

  useEffect(() => {
    if (phase !== "running" && phase !== "queued") return;
    const id = setInterval(() => setElapsed((Date.now() - t0.current) / 1000), 250);
    return () => clearInterval(id);
  }, [phase]);

  const startRun = useCallback(async (c: Component, fw = framework) => {
    if (!paperId) return;
    stopRun();
    setComponent(c); setFramework(fw); setEvents([]); setResult(null); setRunErr(null); setPhase("queued");
    setChat([]); setAblation({ busy: false }); setTranspile({ busy: false });
    t0.current = Date.now(); setElapsed(0);
    try {
      const { run_id } = await api.startRun(paperId, c.id, fw);
      const ctl = { id: run_id, since: 0, stop: false }; runRef.current = ctl;
      while (!ctl.stop) {
        const r = await api.run(ctl.id, ctl.since);
        if (r.events.length) { ctl.since = r.total_events; setEvents(prev => [...prev, ...r.events]); }
        setPhase(r.state === "queued" ? "queued" : r.state === "running" ? "running" : r.state);
        if (r.state === "done" || r.state === "failed") { setResult(r.result); setElapsed((Date.now() - t0.current) / 1000); setTimeout(refreshShelf, 400); break; }
        await new Promise(res => setTimeout(res, 600));
      }
    } catch (e) { setRunErr(errMsg(e)); setPhase("failed"); }
  }, [paperId, framework, refreshShelf]);

  /** Re-open a saved run exactly as it happened (events, code, traces). No API call to the LLM, no GPU. */
  const openSaved = useCallback(async (id: string) => {
    stopRun();
    try {
      const r = await api.historyItem(id);
      const c = components.find(x => x.id === r.meta.target_component) ?? { id: r.meta.target_component, name: r.meta.target_component, description: null, degree: 0, formulas: 0 };
      setComponent(c); setFramework(r.meta.target_framework ?? "PyTorch"); setEvents(r.events); setResult(r.result);
      setPhase(r.state === "done" ? "done" : "failed"); setElapsed(r.events[r.events.length - 1]?.t ?? 0); setRunErr(null);
      setChat([]); setAblation({ busy: false }); setTranspile({ busy: false });
    } catch (e) { setRunErr(errMsg(e)); }
  }, [components]);

  // ---------- ingest ----------
  const upload = useCallback(async (file: File) => {
    setIngestErr(null); setJob(null);
    try {
      const { job_id } = await api.ingest(file);
      for (;;) {
        const j = await api.ingestStatus(job_id); setJob(j);
        if (j.state === "done") { const p = await refreshPapers(); if (j.result) setPaperId(j.result.paper_id); return p; }
        if (j.state === "failed") { setIngestErr(j.error ?? "Ingestion failed"); return; }
        await new Promise(res => setTimeout(res, 1500));
      }
    } catch (e) { setIngestErr(errMsg(e)); }
  }, [refreshPapers]);

  // ---------- derived ----------
  const code = useMemo(() => {
    if (result?.generated_code) return result.generated_code;
    for (let i = events.length - 1; i >= 0; i--) if (events[i].code) return events[i].code!;
    return "";
  }, [events, result]);
  const tags = useMemo(() => parseTags(code), [code]);
  const current = events.length ? events[events.length - 1] : null;
  const activeNode = phase === "running" ? (current?.node === "start" ? "researcher" : current?.node ?? null) : null;
  const litIds = useMemo(() => {
    const s = new Set<string>();
    events.forEach(e => e.node_ids?.forEach(i => s.add(i)));
    tags.forEach(t => s.add(t.nodeId));
    return s;
  }, [events, tags]);
  const attempts = events.filter(e => e.node === "coder").length;
  const traces = result?.tensor_shape_traces ?? {};
  const expectedInputs = useMemo(() => [...events].reverse().find(e => e.expected_inputs)?.expected_inputs ?? {}, [events]);
  const skeleton = useMemo(() => [...events].reverse().find(e => e.skeleton)?.skeleton ?? "", [events]);

  // ---------- explainer / tools ----------
  const ask = useCallback(async (question: string, start: number, end: number) => {
    if (!paperId || !code || !question.trim()) return;
    const history = chat.filter(c => !c.error).map(c => ({ role: c.role, content: c.content }));
    setChat(c => [...c, { role: "user", content: question }]); setAsking(true);
    try {
      const r: ExplainResult = await api.explain({ paper_id: paperId, user_question: question, start, end, code, history });
      setChat(c => [...c, { role: "assistant", content: r.answer_markdown, cites: r.cited_graph_nodes, passages: r.cited_paper_passages }]);
    } catch (e) { setChat(c => [...c, { role: "assistant", content: errMsg(e), error: true }]); }
    finally { setAsking(false); }
  }, [paperId, code, chat]);

  const modelInputs = (traces as any)?.__model__?.input as Record<string, number[]> | undefined;
  const inputsForTools = useMemo<Record<string, string>>(() => {
    if (Object.keys(expectedInputs).length) return expectedInputs;
    return Object.fromEntries(Object.entries(modelInputs ?? {}).map(([k, v]) => [k, `[${v.join(", ")}]`]));
  }, [expectedInputs, modelInputs]);

  const runAblation = useCallback(async (mutation: string) => {
    if (!paperId || !component || !code) return;
    setAblation({ busy: true });
    try { setAblation({ busy: false, data: await api.ablation({ paper_id: paperId, target_component: component.id, baseline_code: code, expected_inputs: inputsForTools, mutation_prompt: mutation }) }); refreshAblations(); }
    catch (e) { setAblation({ busy: false, err: errMsg(e) }); }
  }, [paperId, component, code, inputsForTools, refreshAblations]);

  const runTranspile = useCallback(async (target: string) => {
    if (!code) return;
    setTranspile({ busy: true });
    try { setTranspile({ busy: false, data: await api.transpile({ source_code: code, source_framework: framework, target_framework: target, architecture_skeleton: skeleton, expected_inputs: inputsForTools }) }); }
    catch (e) { setTranspile({ busy: false, err: errMsg(e) }); }
  }, [code, framework, skeleton, inputsForTools]);

  return {
    health, offline, papers, papersLoaded, paperId, setPaperId, graph, components, loadErr, selected, setSelected, hover, setHover,
    shelf, openSaved, refreshShelf, passages, ablations,
    phase, events, current, activeNode, attempts, result, component, framework, setFramework, elapsed, runErr, startRun,
    code, tags, litIds, traces, expectedInputs, skeleton,
    job, ingestErr, upload,
    chat, asking, ask, ablation, runAblation, transpile, runTranspile,
    nodeById: (id: string | null) => (id && graph?.nodes.find(n => n.id === id)) || null,
  };
}
export type Workspace = ReturnType<typeof useWorkspace>;
