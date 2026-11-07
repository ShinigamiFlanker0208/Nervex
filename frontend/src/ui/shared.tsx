// Behaviour shared by all three faces. Markup is neutral (`nx-*` classes); each shell styles it.
import { useMemo, useRef, useState } from "react";
import type { Workspace } from "../data/useWorkspace";
import { SHAPE_RE, tagForLine } from "../lib/codeTags";
import { Markdown } from "../lib/markdown";

export const AGENTS = [
  { key: "researcher", label: "Researcher", verb: "reading the graph" },
  { key: "architect", label: "Architect", verb: "drafting the skeleton" },
  { key: "coder", label: "Coder", verb: "writing the math" },
  { key: "sandbox_executor", label: "Sandbox", verb: "running it for real" },
  { key: "critic", label: "Critic", verb: "diagnosing the failure" },
] as const;

export function describeEvent(e: { node: string; formulas?: number; lines?: number; status?: string; attempt?: number | null; layers?: string[]; feedback?: string; expected_inputs?: Record<string, string> }): string {
  switch (e.node) {
    case "start": return "run started";
    case "researcher": return `pulled ${e.formulas ?? 0} formulas from the graph`;
    case "architect": return `skeleton ready · inputs ${Object.entries(e.expected_inputs ?? {}).map(([k, v]) => `${k}${v}`).join(" ") || "none"}`;
    case "coder": return `wrote ${e.lines ?? 0} lines (attempt ${e.attempt ?? "?"})`;
    case "sandbox_executor": return e.status === "SUCCESS" ? `passed · traced ${e.layers?.length ?? 0} layers` : `failed · ${e.status}`;
    case "critic": return `diagnosis: ${(e.feedback ?? "").replace(/\s+/g, " ").slice(0, 120)}`;
    case "end": return `finished · ${e.status}`;
    default: return e.node;
  }
}

export const NODE_LABEL: Record<string, string> = { start: "start", researcher: "Researcher", architect: "Architect", coder: "Coder", sandbox_executor: "Sandbox", critic: "Critic", end: "end" };

/** Every real event of the run, newest last. Used wherever a face wants a visible audit trail. */
export function EventLog({ ws }: { ws: Workspace }) {
  if (!ws.events.length) return <div className="nx-empty">{ws.phase === "queued" ? "Waiting for the GPU…" : "No events yet."}</div>;
  return (
    <ol className="nx-events">
      {ws.events.map(e => (
        <li key={e.i} className={`ev-${e.node}${e.status && e.status !== "SUCCESS" && e.node === "sandbox_executor" ? " bad" : ""}`}>
          <time>+{e.t.toFixed(1)}s</time><b>{NODE_LABEL[e.node] ?? e.node}</b><span>{describeEvent(e)}</span>
        </li>
      ))}
    </ol>
  );
}

export const fmtSecs = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// ----------------------------------------------------------------------------------------------
export function CodeView({ ws, range, setRange }: { ws: Workspace; range: [number, number] | null; setRange: (r: [number, number] | null) => void }) {
  const lines = useMemo(() => ws.code.split("\n"), [ws.code]);
  const anchor = useRef<number | null>(null);
  const hoverTag = ws.hover;
  const nameOf = (id: string | null) => ws.nodeById(id)?.name ?? id ?? "";
  return (
    <div className="nx-code" role="region" aria-label="Generated code">
      {lines.map((l, i) => {
        const n = i + 1;
        const tag = tagForLine(ws.tags, lines, n);
        const isTagLine = ws.tags.some(t => t.line === n);
        const cls = ["nx-line",
          isTagLine && "is-tag", SHAPE_RE.test(l) && "is-shape", /^\s*#/.test(l) && "is-comment",
          tag && hoverTag === tag && "is-lit", range && n >= range[0] && n <= range[1] && "is-sel"].filter(Boolean).join(" ");
        return (
          <div key={i} className={cls}
            onMouseEnter={() => tag && ws.setHover(tag)} onMouseLeave={() => tag && ws.setHover(null)}
            onClick={e => {
              if (e.shiftKey && anchor.current) setRange([Math.min(anchor.current, n), Math.max(anchor.current, n)]);
              else { anchor.current = n; setRange([n, n]); }
            }}>
            <span className="nx-ln">{n}</span>
            <span className="nx-src">{l || " "}</span>
            {tag && <span className="nx-tagname" title={tag}>{nameOf(tag)}</span>}
          </div>
        );
      })}
    </div>
  );
}

// ----------------------------------------------------------------------------------------------
export function ChatPane({ ws, range }: { ws: Workspace; range: [number, number] | null }) {
  const [q, setQ] = useState("");
  const total = ws.code.split("\n").length;
  const [s, e] = range ?? [1, total];
  const send = (text: string) => { if (text.trim()) { ws.ask(text, s, e); setQ(""); } };
  const suggestions = ["Why is this implemented this way?", "Which equation does this correspond to?", "What are the tensor shapes here?"];
  return (
    <div className="nx-chat">
      <div className="nx-chat-scope">asking about <b>{range ? `lines ${s}–${e}` : "the whole module"}</b> <span className="nx-dim">· click / shift-click lines to narrow</span></div>
      <div className="nx-chat-log">
        {ws.chat.length === 0 && <div className="nx-empty">{ws.code ? "Ask anything about this code. Answers are grounded in the paper's own graph and text." : "Generate a component first."}</div>}
        {ws.chat.map((t, i) => (
          <div key={i} className={`nx-turn ${t.role}${t.error ? " err" : ""}`}>
            {t.role === "assistant" && !t.error ? <Markdown text={t.content} /> : <p>{t.content}</p>}
            {!!t.cites?.length && <div className="nx-cites">{t.cites.map(c => <button key={c} className="nx-chip" onMouseEnter={() => ws.setHover(c)} onMouseLeave={() => ws.setHover(null)} onClick={() => ws.setSelected(c)}>{ws.nodeById(c)?.name ?? c}</button>)}</div>}
          </div>
        ))}
        {ws.asking && <div className="nx-turn assistant nx-thinking"><span /><span /><span /></div>}
      </div>
      {ws.code && ws.chat.length === 0 && <div className="nx-suggest">{suggestions.map(x => <button key={x} className="nx-chip" onClick={() => send(x)}>{x}</button>)}</div>}
      <form onSubmit={ev => { ev.preventDefault(); send(q); }} className="nx-ask">
        <input value={q} onChange={ev => setQ(ev.target.value)} placeholder={ws.code ? "Ask about the selected lines…" : "No code yet"} disabled={!ws.code || ws.asking} />
        <button disabled={!ws.code || ws.asking || !q.trim()}>Ask</button>
      </form>
    </div>
  );
}

// ----------------------------------------------------------------------------------------------
export function ToolsPane({ ws }: { ws: Workspace }) {
  const [mut, setMut] = useState("");
  const [showCode, setShowCode] = useState(false);
  const other = ws.framework === "PyTorch" ? "JAX" : "PyTorch";
  const t = ws.transpile.data, a = ws.ablation.data;
  const layers = (m: any) => m?.traced_layers ?? m?.layers?.length ?? "–";
  return (
    <div className="nx-tools">
      <section>
        <h4>Ablate</h4>
        <p className="nx-dim">Describe a change. Nervex rewrites the module, re-runs it in the sandbox and compares.</p>
        <form onSubmit={e => { e.preventDefault(); ws.runAblation(mut); }} className="nx-ask">
          <input value={mut} onChange={e => setMut(e.target.value)} placeholder="e.g. remove the output projection" disabled={!ws.code || ws.ablation.busy} />
          <button disabled={!ws.code || ws.ablation.busy || !mut.trim()}>{ws.ablation.busy ? "Running…" : "Run"}</button>
        </form>
        {ws.ablation.err && <div className="nx-err">{ws.ablation.err}</div>}
        {a && <div className="nx-result">
          <div className="nx-compare"><div><span className="nx-dim">baseline layers</span><b>{layers(a.baseline_metrics)}</b></div><div><span className="nx-dim">ablated layers</span><b>{layers(a.ablated_metrics)}</b></div></div>
          <Markdown text={a.diff_explanation} />
        </div>}
      </section>
      <section>
        <h4>Transpile</h4>
        <p className="nx-dim">{ws.framework} → {other}. The result is executed in the sandbox, not just parsed.</p>
        <button className="nx-btn" disabled={!ws.code || ws.transpile.busy} onClick={() => ws.runTranspile(other)}>{ws.transpile.busy ? "Translating & running…" : `Translate to ${other}`}</button>
        {ws.transpile.err && <div className="nx-err">{ws.transpile.err}</div>}
        {t && <div className="nx-result">
          <div className="nx-badges">
            <span className={`nx-badge ${t.syntax_valid ? "ok" : "bad"}`}>{t.syntax_valid ? "compiles" : "does not compile"}</span>
            <span className={`nx-badge ${t.runtime_verified ? "ok" : t.runtime_verified === false ? "bad" : "warn"}`}>
              {t.runtime_verified ? "runs in sandbox" : t.runtime_verified === false ? "fails when run" : "not executed"}
            </span>
            {t.output_shape != null && <span className="nx-badge">out {JSON.stringify(t.output_shape)}</span>}
          </div>
          {t.notes && <Markdown text={t.notes.slice(0, 900)} />}
          <button className="nx-link" onClick={() => setShowCode(s => !s)}>{showCode ? "hide code" : "show code"}</button>
          {showCode && <pre className="nx-pre">{t.transpiled_code}</pre>}
          <p className="nx-dim">Passing proves it runs and returns finite numbers of the right shape, not that the math matches.</p>
        </div>}
      </section>
    </div>
  );
}

// ----------------------------------------------------------------------------------------------
export function IngestPanel({ ws }: { ws: Workspace }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const busy = ws.job && (ws.job.state === "queued" || ws.job.state === "running");
  const take = (f?: File) => { if (f) ws.upload(f); };
  return (
    <div className={`nx-drop${over ? " over" : ""}${busy ? " busy" : ""}`}
      onDragOver={e => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}>
      <input ref={input} type="file" accept="application/pdf" hidden onChange={e => take(e.target.files?.[0])} />
      {!ws.job && <>
        <div className="nx-drop-title">Feed it a paper</div>
        <div className="nx-dim">Drop a PDF here. OCR, equation masking and graph extraction take minutes; you can watch every step.</div>
        <button className="nx-btn" onClick={() => input.current?.click()}>Choose a PDF</button>
      </>}
      {ws.job && <div className="nx-progress">
        <div className="nx-drop-title">{ws.job.state === "failed" ? "Ingestion failed" : ws.job.state === "done" ? "Ingested" : "Ingesting " + ws.job.filename}</div>
        <ol>{ws.job.progress.slice(-6).map((p, i, arr) => <li key={i} className={i === arr.length - 1 && busy ? "live" : ""}>{p}</li>)}</ol>
        {ws.job.state === "done" && ws.job.result && <div className="nx-dim">{ws.job.result.num_entities_extracted} entities · {ws.job.result.num_relationships_merged} relationships</div>}
      </div>}
      {ws.ingestErr && <div className="nx-err">{ws.ingestErr}</div>}
    </div>
  );
}

export function OfflineNotice({ ws }: { ws: Workspace }) {
  if (!ws.offline) return null;
  return <div className="nx-offline"><b>API unreachable.</b> Start it with <code>uv run python main.py</code> (and Neo4j: <code>docker start nervex-neo4j</code>).</div>;
}

export function Telemetry({ ws }: { ws: Workspace }) {
  const entries = Object.entries(ws.traces).filter(([k]) => k !== "__model__");
  const model = (ws.traces as any).__model__;
  const fmt = (s: any): string => (Array.isArray(s) ? `[${s.map(fmt).join(", ")}]` : typeof s === "object" && s ? JSON.stringify(s) : String(s));
  if (!entries.length && !model) return <div className="nx-empty">Shape telemetry appears after a run passes the sandbox.</div>;
  return (
    <div className="nx-telemetry">
      {model && <div className="nx-trace model"><span className="nx-trace-name">model</span><span className="nx-trace-shape">{Object.values(model.input ?? {}).map(fmt).join(" , ")} → {fmt(model.output)}</span></div>}
      {entries.map(([k, v]: [string, any]) => (
        <div className="nx-trace" key={k} onMouseEnter={() => ws.setHover(null)}>
          <span className="nx-trace-name">{k}</span>
          <span className="nx-trace-shape">{fmt(v.input_shape ?? v.input ?? "?")} → {fmt(v.output_shape ?? v.output ?? "?")}</span>
        </div>
      ))}
    </div>
  );
}
