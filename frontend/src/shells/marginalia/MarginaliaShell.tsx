import { useEffect, useMemo, useState } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { ChatPane, CodeView, IngestPanel, OfflineNotice, Telemetry, ToolsPane, fmtSecs } from "../../ui/shared";
import { Markdown } from "../../lib/markdown";
import { Constellation } from "./Constellation";
import "./marginalia.css";

type Tab = "margin" | "ask" | "appendix";
const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"];

interface Draft { no: number; verdict: "pending" | "passed" | "rejected"; note?: string; lines?: number; log?: string }
function toDrafts(ws: Workspace): Draft[] {
  const out: Draft[] = [];
  for (const e of ws.events) {
    if (e.node === "coder") out.push({ no: out.length + 1, verdict: "pending", lines: e.lines });
    else if (e.node === "sandbox_executor" && out.length) { const d = out[out.length - 1]; d.verdict = e.status === "SUCCESS" ? "passed" : "rejected"; d.log = e.log_tail; }
    else if (e.node === "critic" && out.length) out[out.length - 1].note = e.feedback;
  }
  return out;
}

export function MarginaliaShell({ ws, faces }: { ws: Workspace; faces: React.ReactNode }) {
  const [tab, setTab] = useState<Tab>("margin");
  const [range, setRange] = useState<[number, number] | null>(null);
  const [adding, setAdding] = useState(false);
  const [wipe, setWipe] = useState(false);
  const drafts = useMemo(() => toDrafts(ws), [ws.events]);                // eslint-disable-line react-hooks/exhaustive-deps
  const paper = ws.papers.find(p => p.paper_id === ws.paperId);
  const focus = ws.nodeById(ws.hover ?? ws.selected);
  const noted = useMemo(() => [...new Set(ws.tags.map(t => t.nodeId))], [ws.tags]);
  const busy = ws.phase === "running" || ws.phase === "queued";
  useEffect(() => { if (ws.phase === "done") { setWipe(true); const t = setTimeout(() => setWipe(false), 1800); return () => clearTimeout(t); } }, [ws.phase]);

  return (
    <div className="mg" data-shell="marginalia">
      <header className="mg-mast">
        <div className="mg-brand"><span className="mg-smallcaps">Nervex</span><i>a reading of</i></div>
        <h1>{paper?.title ?? (ws.offline ? "The API is asleep" : "An empty shelf")}</h1>
        <div className="mg-meta">
          {paper && <span>{paper.nodes} entities · {paper.edges} relations</span>}
          {ws.papers.length > 1 && <select value={ws.paperId ?? ""} onChange={e => ws.setPaperId(e.target.value)}>{ws.papers.map(p => <option key={p.paper_id} value={p.paper_id}>{p.title}</option>)}</select>}
          <button className="mg-link" onClick={() => setAdding(a => !a)}>add a paper</button>
          {ws.health?.llm.provider && <span className="mg-pill">{ws.health.llm.provider}</span>}
          {faces}
        </div>
      </header>
      <OfflineNotice ws={ws} />
      {(adding || (!ws.papers.length && !ws.offline)) && <section className="mg-sheet"><IngestPanel ws={ws} /></section>}

      <div className="mg-grid">
        <aside className="mg-contents">
          <div className="mg-fig"><Constellation ws={ws} /><figcaption>Fig. 1 · the paper's most connected ideas. <b>Red</b> marks what the code is built from.</figcaption></div>
          <h2>Chapters</h2>
          <div className="mg-fw">{["PyTorch", "JAX"].map(f => <button key={f} className={ws.framework === f ? "on" : ""} onClick={() => ws.setFramework(f)}>{f}</button>)}</div>
          <ol className="mg-toc">
            {ws.components.map((c, i) => (
              <li key={c.id} className={ws.component?.id === c.id ? "cur" : ""} onMouseEnter={() => ws.setHover(c.id)} onMouseLeave={() => ws.setHover(null)}>
                <button disabled={busy} onClick={() => ws.startRun(c)}><span className="n">{i + 1}.</span><span className="t">{c.name ?? c.id}</span><span className="lead" /><span className="p" title="formulas the code can cite">{c.formulas ? `${c.formulas}ƒ` : c.degree}</span></button>
              </li>
            ))}
          </ol>
          {!ws.components.length && <p className="mg-dim">{ws.paperId ? "No chapters found." : "Add a paper to begin."}</p>}
        </aside>

        <main className="mg-ms">
          {ws.phase === "idle" && (
            <div className="mg-open">
              <p className="mg-kicker">{focus ? (focus.type ?? "entity") : "begin"}</p>
              <h2 className="mg-big">{focus ? focus.name : "Choose a chapter."}</h2>
              <p className="mg-lede">{focus?.description ?? "Pick a chapter on the left. Nervex will read what the paper says about it, draft the code, run it for real in a sandbox, and show you every draft, including the ones it threw away."}</p>
            </div>
          )}
          {ws.phase !== "idle" && <>
            <div className="mg-run">
              <h2 className="mg-big sm">{ws.component?.name}<small> · {ws.framework}</small></h2>
              <span className={`mg-stamp ${ws.phase}`}>{ws.phase === "done" ? "verified" : ws.phase === "failed" ? "not verified" : ws.phase === "queued" ? "waiting for the GPU" : `drafting · ${fmtSecs(ws.elapsed)}`}</span>
            </div>
            {ws.runErr && <div className="nx-err">{ws.runErr}</div>}
            <ol className="mg-drafts">
              {drafts.map(d => (
                <li key={d.no} className={d.verdict}>
                  <span className="mg-folio">Draft {ROMAN[d.no] ?? d.no}</span>
                  <span className="mg-v">{d.verdict === "pending" ? "being tested…" : d.verdict === "passed" ? "ran cleanly in the sandbox" : "rejected by the sandbox"}{d.lines ? ` · ${d.lines} lines` : ""}</span>
                  {d.note && <blockquote className="mg-pencil"><b>Critic</b><Markdown text={d.note.slice(0, 520)} /></blockquote>}
                  {d.verdict === "rejected" && d.log && <details><summary>what the sandbox said</summary><pre>{d.log.slice(-900)}</pre></details>}
                </li>
              ))}
              {busy && !drafts.length && <li className="pending"><span className="mg-folio">Researching</span><span className="mg-v">reading the paper's graph{ws.events.length > 1 ? ` · ${ws.events[ws.events.length - 1].node}` : ""}…</span></li>}
            </ol>
            {ws.code && <div className="mg-codebox"><CodeView ws={ws} range={range} setRange={setRange} /></div>}
            {ws.phase === "failed" && ws.result && <p className="mg-dim">No draft passed after {ws.result.attempt_count} attempt(s){ws.result.error_details ? ` (${ws.result.error_details})` : ""}. The last draft is shown above; it is not verified.</p>}
            {ws.phase === "done" && <section className={`mg-plate${wipe ? " wipe" : ""}`}><h3>Plate II <span>tensor flow, measured by hooks inside the sandbox</span></h3><Telemetry ws={ws} /></section>}
          </>}
        </main>

        <aside className="mg-margin">
          <nav className="mg-tabs">{(["margin", "ask", "appendix"] as Tab[]).map(t => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t === "ask" ? "ask the margin" : t}</button>)}</nav>
          {tab === "margin" && <div className="mg-notes">
            {focus && <article className="hot"><small>{focus.type}</small><h4>{focus.name}</h4><p>{focus.description}</p></article>}
            {noted.length === 0 && !focus && <p className="mg-dim">Marginal notes appear here: every <code># Implements:</code> tag in the code links to the paper concept it came from.</p>}
            {noted.map(id => { const n = ws.nodeById(id); return (
              <article key={id} className={ws.hover === id ? "hot" : ""} onMouseEnter={() => ws.setHover(id)} onMouseLeave={() => ws.setHover(null)}>
                <small>{n?.type ?? "node"} · {id}</small><h4>{n?.name ?? id}</h4>{n?.description ? <p>{n.description}</p> : <p className="mg-dim">Not in the top of the graph.</p>}
              </article>); })}
          </div>}
          {tab === "ask" && <ChatPane ws={ws} range={range} />}
          {tab === "appendix" && <ToolsPane ws={ws} />}
        </aside>
      </div>
    </div>
  );
}
