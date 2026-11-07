import { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import type { Workspace } from "../../data/useWorkspace";
import { diffLines, hunks } from "../../lib/diff";
import { Markdown } from "../../lib/markdown";
import { ChatPane, CodeView, IngestPanel, OfflineNotice, ToolsPane, fmtSecs } from "../../ui/shared";
import { Constellation } from "../marginalia/Constellation";
import "../marginalia/marginalia.css";
import "./reading.css";

const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"];
const RAIL = [["researcher", "reads"], ["architect", "drafts the contract"], ["coder", "writes"], ["sandbox_executor", "runs it"], ["critic", "objects"]] as const;
const NAME: Record<string, string> = { researcher: "Researcher", architect: "Architect", coder: "Coder", sandbox_executor: "Sandbox", critic: "Critic" };

interface Draft { no: number; code: string; verdict: "pending" | "passed" | "rejected"; note?: string; log?: string }
function drafts(ws: Workspace): Draft[] {
  const out: Draft[] = [];
  for (const e of ws.events) {
    if (e.node === "coder") out.push({ no: out.length + 1, code: e.code ?? "", verdict: "pending" });
    else if (e.node === "sandbox_executor" && out.length) { const d = out[out.length - 1]; d.verdict = e.status === "SUCCESS" ? "passed" : "rejected"; d.log = e.log_tail; }
    else if (e.node === "critic" && out.length) out[out.length - 1].note = e.feedback;
  }
  return out;
}

function Tex({ src }: { src: string }) {
  const html = useMemo(() => {
    const t = src.trim().replace(/^\$+|\$+$/g, "");
    try { return katex.renderToString(t, { throwOnError: false, displayMode: false, trust: false }); } catch { return null; }
  }, [src]);
  return html ? <span className="rr-tex" dangerouslySetInnerHTML={{ __html: html }} /> : <code>{src}</code>;
}

function Diff({ a, b }: { a: string; b: string }) {
  const h = useMemo(() => hunks(diffLines(a, b)), [a, b]);
  return <div className="rr-diff">{h.map((x, i) => x.kind === "fold"
    ? <div key={i} className="fold">· · · {x.n} unchanged line{x.n === 1 ? "" : "s"} · · ·</div>
    : <div key={i} className={x.kind}><span>{x.kind === "add" ? "+" : x.kind === "del" ? "−" : " "}</span>{x.text || " "}</div>)}</div>;
}

/** The live agent rail: who is working, in words, on a pencil line. */
function Rail({ ws }: { ws: Workspace }) {
  const visited = new Set(ws.events.map(e => e.node));
  const d = ws.events.filter(e => e.node === "coder").length;
  const act = ws.activeNode;
  const say = ws.phase === "queued" ? "Waiting for the GPU. One sandbox at a time." :
    ws.phase === "done" ? `Verified. ${d} draft${d === 1 ? "" : "s"}, ${fmtSecs(ws.elapsed)}.` :
    ws.phase === "failed" ? "The loop ended without a passing draft." :
    act === "researcher" ? "The Researcher is reading the paper's graph…" :
    act === "architect" ? "The Architect is drawing the shape contract…" :
    act === "coder" ? `The Coder is writing draft ${ROMAN[d + 1] ?? d + 1}…` :
    act === "sandbox_executor" ? `The sandbox is running draft ${ROMAN[d] ?? d}…` :
    act === "critic" ? `The Critic is reading what broke in draft ${ROMAN[d] ?? d}…` : "…";
  return (
    <div className={`rr-rail ${ws.phase}`}>
      <ol>{RAIL.map(([k, verb]) => <li key={k} className={`${act === k ? "act " : ""}${visited.has(k) ? "seen " : ""}${k === "critic" ? "crit" : ""}`}><b>{NAME[k]}</b><i>{verb}</i></li>)}</ol>
      <p><span>{say}</span>{(ws.phase === "running" || ws.phase === "queued") && <time>{fmtSecs(ws.elapsed)}</time>}</p>
    </div>
  );
}

export function ReadingRoom({ ws, faces, onOpenInstrument, onTour }: { ws: Workspace; faces: React.ReactNode; onOpenInstrument: () => void; onTour: () => void }) {
  const [tab, setTab] = useState<"margin" | "ask" | "appendix">("margin");
  const [range, setRange] = useState<[number, number] | null>(null);
  const [adding, setAdding] = useState(false);
  const [openDiff, setOpenDiff] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const codeBox = useRef<HTMLDivElement>(null);
  const D = useMemo(() => drafts(ws), [ws.events]);                                     // eslint-disable-line react-hooks/exhaustive-deps
  const paper = ws.papers.find(p => p.paper_id === ws.paperId);
  const focus = ws.nodeById(ws.hover ?? ws.selected);
  const busy = ws.phase === "running" || ws.phase === "queued";
  const eqs = useMemo(() => [...ws.events].reverse().find(e => e.formulas_tex)?.formulas_tex ?? [], [ws.events]);
  const tagLine = (id: string) => ws.tags.find(t => t.nodeId === id)?.line;
  const goLine = (n: number) => { const el = codeBox.current?.querySelectorAll(".nx-line")[n - 1] as HTMLElement | undefined; el?.scrollIntoView({ behavior: "smooth", block: "center" }); setRange([n, n + 1]); };
  const fname = `${(ws.component?.id ?? "module").replace(/[^a-z0-9_]/gi, "_")}_${ws.framework.toLowerCase()}.py`;
  const download = () => { const u = URL.createObjectURL(new Blob([ws.code], { type: "text/x-python" })); const a = document.createElement("a"); a.href = u; a.download = fname; a.click(); URL.revokeObjectURL(u); };
  const copy = async () => { try { await navigator.clipboard.writeText(ws.code); setCopied(true); setTimeout(() => setCopied(false), 1400); } catch { /* clipboard blocked */ } };
  useEffect(() => { setOpenDiff(null); }, [ws.component?.id]);

  return (
    <div className="mg rr" data-shell="reading">
      <header className="mg-mast">
        <div className="mg-brand"><span className="mg-smallcaps">Nervex</span><i>a reading of</i></div>
        <h1>{paper?.title ?? (ws.offline ? "The API is asleep" : "An empty shelf")}</h1>
        <div className="mg-meta">
          {paper && <span>{paper.nodes} entities · {paper.edges} relations · {ws.shelf.length} run{ws.shelf.length === 1 ? "" : "s"} on the shelf</span>}
          {ws.papers.length > 1 && <select value={ws.paperId ?? ""} onChange={e => ws.setPaperId(e.target.value)}>{ws.papers.map(p => <option key={p.paper_id} value={p.paper_id}>{p.title}</option>)}</select>}
          <button className="mg-link" onClick={() => setAdding(a => !a)}>add a paper</button>
          <button className="mg-link" onClick={onTour}>how it works</button>
          {ws.health?.llm.provider && <span className="mg-pill">{ws.health.llm.provider}</span>}
          {faces}
        </div>
      </header>
      <OfflineNotice ws={ws} />
      {(adding || (!ws.papers.length && !ws.offline)) && <section className="mg-sheet"><IngestPanel ws={ws} /></section>}

      <div className="mg-grid">
        <aside className="mg-contents">
          <h2>Chapters</h2>
          <div className="mg-fw">{["PyTorch", "JAX"].map(f => <button key={f} className={ws.framework === f ? "on" : ""} onClick={() => ws.setFramework(f)}>{f}</button>)}</div>
          <ol className="mg-toc">{ws.components.map((c, i) => {
            const dup = ws.components.filter(x => (x.name ?? x.id) === (c.name ?? c.id)).length > 1;
            const done = ws.shelf.find(r => r.component === c.id && r.status === "SUCCESS");
            return <li key={c.id} className={ws.component?.id === c.id ? "cur" : ""} onMouseEnter={() => ws.setHover(c.id)} onMouseLeave={() => ws.setHover(null)}>
              <button disabled={busy} onClick={() => ws.startRun(c)} title={c.description ?? ""}><span className="n">{i + 1}.</span>
                <span className="t">{c.name ?? c.id}{dup && <small className="rr-id"> {c.id}</small>}{done && <em className="rr-tick" title="verified before">✓</em>}</span><span className="lead" /><span className="p">{c.formulas ? `${c.formulas}ƒ` : c.degree}</span></button></li>; })}
          </ol>
          {!ws.components.length && <p className="mg-dim">{ws.paperId ? "No chapters found." : "Add a paper to begin."}</p>}
          {ws.phase !== "idle" && <div className="mg-fig rr-mini"><Constellation ws={ws} /><figcaption>Red: what this code is built from.</figcaption></div>}
        </aside>

        <main className="mg-ms">
          {ws.phase === "idle" ? <>
            <section className="rr-front">
              <div className="rr-front-plate"><Constellation ws={ws} /></div>
              <div className="rr-front-cap">
                <p className="mg-kicker">{focus ? focus.type : "frontispiece"}</p>
                <h2 className="mg-big">{focus ? focus.name : <>The paper, as a <em>graph</em>.</>}</h2>
                <p className="mg-lede">{focus?.description ?? `${paper ? `${Math.min(paper.nodes, 140)} of ${paper.nodes} entities, the best connected.` : ""} Hover a chapter on the left to find it here. Pick one to have it built, run and verified.`}</p>
              </div>
            </section>
            <section className="rr-shelf">
              <h3>The shelf <span>every run is kept, passed or not</span></h3>
              {!ws.shelf.length && <p className="mg-dim">Empty. Your first run will be filed here.</p>}
              <div className="rr-cards">{ws.shelf.slice(0, 12).map(r => (
                <button key={r.run_id} className={`rr-card ${r.status === "SUCCESS" ? "ok" : "bad"}`} onClick={() => ws.openSaved(r.run_id)}>
                  <span className="rr-card-st">{r.status === "SUCCESS" ? "verified" : "not verified"}</span>
                  <b>{ws.components.find(c => c.id === r.component)?.name ?? r.component}</b>
                  <i>{r.framework} · {r.attempts} draft{r.attempts === 1 ? "" : "s"} · {fmtSecs(r.duration)}{r.layers ? ` · ${r.layers} layers` : ""}</i>
                  <small>{r.created_at ? new Date(r.created_at * 1000).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : ""}</small>
                </button>))}</div>
            </section>
          </> : <>
            <div className="mg-run">
              <h2 className="mg-big sm">{ws.component?.name}<small> · {ws.framework}</small></h2>
              <span className={`mg-stamp ${ws.phase}`}>{ws.phase === "done" ? "verified" : ws.phase === "failed" ? "not verified" : ws.phase === "queued" ? "waiting" : "drafting"}</span>
            </div>
            <Rail ws={ws} />
            {ws.runErr && <div className="nx-err">{ws.runErr}</div>}
            <ol className="mg-drafts">{D.map((d, i) => (
              <li key={d.no} className={d.verdict}>
                <div className="rr-draft-h"><span className="mg-folio">Draft {ROMAN[d.no] ?? d.no}</span>
                  <span className="mg-v">{d.verdict === "pending" ? "being tested…" : d.verdict === "passed" ? "ran cleanly in the sandbox" : "rejected by the sandbox"} · {d.code.split("\n").length} lines</span>
                  {i > 0 && d.code && D[i - 1].code && <button className="rr-difbtn" onClick={() => setOpenDiff(o => o === i ? null : i)}>{openDiff === i ? "hide changes" : (() => { const x = diffLines(D[i - 1].code, d.code); return `what changed · +${x.filter(l => l.kind === "add").length} −${x.filter(l => l.kind === "del").length}`; })()}</button>}</div>
                {d.note && <blockquote className="mg-pencil"><b>Critic</b><Markdown text={d.note.slice(0, 520)} /></blockquote>}
                {d.verdict === "rejected" && d.log && <details><summary>what the sandbox said</summary><pre>{d.log.slice(-900)}</pre></details>}
                {openDiff === i && <Diff a={D[i - 1].code} b={d.code} />}
              </li>))}
              {busy && !D.length && <li className="pending"><span className="mg-folio">Reading</span><span className="mg-v">no draft yet</span></li>}
            </ol>
            {ws.code && <div className="rr-codebar"><span>{fname}</span><button onClick={copy}>{copied ? "copied" : "copy"}</button><button onClick={download}>download .py</button></div>}
            {ws.code && <div className="mg-codebox" ref={codeBox}><CodeView ws={ws} range={range} setRange={setRange} /></div>}
            {ws.phase === "failed" && ws.result && <p className="mg-dim">No draft passed after {ws.result.attempt_count} attempt(s){ws.result.error_details ? ` (${ws.result.error_details})` : ""}. The last draft is shown; it is not verified.</p>}
            {ws.events.length > 0 && !busy && <button className={`mg-gate${ws.phase === "done" ? " ok" : ""}`} onClick={onOpenInstrument} disabled={busy}>
              <small>Plate II</small><span>{ws.phase === "done" ? "Enter the instrument room" : "Open the instrument room"}</span><i>measured in the sandbox, not guessed →</i></button>}
          </>}
        </main>

        <aside className="mg-margin">
          <nav className="mg-tabs">{(["margin", "ask", "appendix"] as const).map(t => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t === "ask" ? "ask the margin" : t}</button>)}</nav>
          {tab === "margin" && <div className="mg-notes">
            {focus && <article className="hot"><small>{focus.type}</small><h4>{focus.name}</h4><p>{focus.description}</p></article>}
            {eqs.length > 0 && <section className="rr-lens"><h5>The equations it was given</h5>
              {eqs.map(q => { const ln = tagLine(q.id); return (
                <div key={q.id} className={`rr-eq${ln ? " cited" : ""}${ws.hover === q.id ? " hot" : ""}`} onMouseEnter={() => ws.setHover(q.id)} onMouseLeave={() => ws.setHover(null)}>
                  <Tex src={q.latex} /><small>{ws.nodeById(q.id)?.name ?? q.id}</small>
                  {ln ? <button onClick={() => goLine(ln)}>cited at line {ln} →</button> : <em>not cited in the code</em>}
                </div>); })}</section>}
            {!eqs.length && !focus && <p className="mg-dim">{ws.phase === "idle" ? "Margin notes appear here: the equations a run was given, and where in the code each one is used." : "No equations reached this run."}</p>}
          </div>}
          {tab === "ask" && <ChatPane ws={ws} range={range} />}
          {tab === "appendix" && <ToolsPane ws={ws} />}
        </aside>
      </div>
    </div>
  );
}
