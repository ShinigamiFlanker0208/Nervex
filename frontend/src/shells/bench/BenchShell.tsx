import { useEffect, useMemo, useState } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { AGENTS, ChatPane, CodeView, EventLog, IngestPanel, OfflineNotice, ToolsPane, describeEvent, fmtSecs } from "../../ui/shared";
import { Radar } from "./Radar";
import "./bench.css";

type Tab = "events" | "ask" | "tools";
const nums = (s: unknown): number[] => (typeof s === "string" ? (s.match(/\d+/g) ?? []).map(Number) : Array.isArray(s) ? (s as number[]) : []);

function Ribbon({ dims, ref0 }: { dims: number[]; ref0?: number[] }) {
  return <div className="bn-rib">{dims.map((d, i) => <span key={i} style={{ flexGrow: Math.log2(d + 1) + .6 }} className={ref0 && ref0[i] !== d ? "chg" : ""}>{d}</span>)}</div>;
}

export function BenchShell({ ws, faces }: { ws: Workspace; faces: React.ReactNode }) {
  const [tab, setTab] = useState<Tab>("events");
  const [range, setRange] = useState<[number, number] | null>(null);
  const [adding, setAdding] = useState(false);
  const [cur, setCur] = useState(0);
  const [now, setNow] = useState(new Date());
  const busy = ws.phase === "running" || ws.phase === "queued";
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(t); }, []);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") { if (e.key === "Escape") (e.target as HTMLElement).blur(); return; }
      if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); setCur(c => Math.min(ws.components.length - 1, c + 1)); }
      if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); setCur(c => Math.max(0, c - 1)); }
      if (e.key === "Enter" && !busy && ws.components[cur]) ws.startRun(ws.components[cur]);
      if (e.key === "f") ws.setFramework(ws.framework === "PyTorch" ? "JAX" : "PyTorch");
      if (e.key === "/") { e.preventDefault(); setTab("ask"); setTimeout(() => document.querySelector<HTMLInputElement>(".bn .nx-ask input")?.focus(), 30); }
    };
    addEventListener("keydown", h); return () => removeEventListener("keydown", h);
  }, [ws, cur, busy]);

  // per-agent stats from REAL timestamps: an event's cost = time since the previous event
  const chain = useMemo(() => AGENTS.map(a => {
    const calls: { dt: number; ev: typeof ws.events[number] }[] = [];
    ws.events.forEach((e, i) => { if (e.node === a.key) calls.push({ dt: Math.max(0, e.t - (ws.events[i - 1]?.t ?? 0)), ev: e }); });
    return { ...a, calls };
  }), [ws.events]);                                                 // eslint-disable-line react-hooks/exhaustive-deps
  const maxDt = Math.max(1, ...chain.flatMap(c => c.calls.map(x => x.dt)));
  const modelT = (ws.traces as any).__model__;
  const layers = Object.entries(ws.traces).filter(([k]) => k !== "__model__") as [string, any][];
  const focus = ws.nodeById(ws.hover ?? ws.selected);
  const led = (key: string) => ws.activeNode === key ? "act" : chain.find(c => c.key === key)?.calls.some(c => c.ev.status && c.ev.status !== "SUCCESS" && key === "sandbox_executor") && ws.phase === "failed" ? "bad" : (chain.find(c => c.key === key)?.calls.length ? "ok" : "");

  return (
    <div className="bn" data-shell="bench">
      <header className="bn-top">
        <b className="bn-logo">NERVEX<i>/bench</i></b>
        <select value={ws.paperId ?? ""} onChange={e => ws.setPaperId(e.target.value)}>{ws.papers.length === 0 && <option>— no paper —</option>}{ws.papers.map(p => <option key={p.paper_id} value={p.paper_id}>{p.title}</option>)}</select>
        <button onClick={() => setAdding(a => !a)}>+ INGEST</button>
        <div className="bn-leds">{ws.health && Object.entries(ws.health.subsystems).map(([k, v]) => <span key={k} className={v ? "ok" : "bad"}>{k.replace("_", "-")}</span>)}
          {ws.health?.llm && <span className="llm">{ws.health.llm.provider}:{(ws.health.llm.model ?? "").split("/").pop()}</span>}</div>
        <span className="bn-grow" /><span className="bn-clock">{now.toTimeString().slice(0, 8)}</span>{faces}
      </header>
      <OfflineNotice ws={ws} />
      {(adding || (!ws.papers.length && !ws.offline)) && <div className="bn-modal"><IngestPanel ws={ws} /></div>}

      <div className="bn-main">
        <section className="bn-mod bn-targets"><h3>TARGETS <small>↑↓ select · ⏎ run · f framework</small></h3>
          <div className="bn-fw">{["PyTorch", "JAX"].map(f => <button key={f} className={ws.framework === f ? "on" : ""} onClick={() => ws.setFramework(f)}>{f.toUpperCase()}</button>)}</div>
          <ul>{ws.components.map((c, i) => (
            <li key={c.id} className={(i === cur ? "cur " : "") + (ws.component?.id === c.id ? "run" : "")} onMouseEnter={() => { setCur(i); ws.setHover(c.id); }} onMouseLeave={() => ws.setHover(null)}>
              <button disabled={busy} onClick={() => ws.startRun(c)}><em>{String(i + 1).padStart(2, "0")}</em><span>{c.name ?? c.id}</span><small title="formulas the code can cite">{c.formulas ? `${c.formulas}ƒ` : c.degree}</small></button>
            </li>))}</ul>
          {!ws.components.length && <p className="bn-dim">{ws.paperId ? "no components" : "ingest a paper"}</p>}
        </section>

        <div className="bn-center">
          <section className="bn-chain">
            {chain.map((c, i) => (
              <div key={c.key} className={`bn-rack ${led(c.key)}`}>
                <div className="bn-rack-h"><i className="bn-led" /><b>{c.label.toUpperCase()}</b><small>×{c.calls.length}</small></div>
                <div className="bn-bars">{c.calls.slice(-9).map((x, j) => <span key={j} style={{ height: `${10 + 90 * x.dt / maxDt}%` }} title={`${x.dt.toFixed(1)}s`} />)}{!c.calls.length && <em>idle</em>}</div>
                <div className="bn-rack-o">{c.calls.length ? describeEvent(c.calls[c.calls.length - 1].ev) : c.verb}</div>
                {i < AGENTS.length - 1 && <u className={ws.activeNode === AGENTS[i + 1].key ? "flow" : ""} />}
              </div>))}
          </section>

          <div className="bn-split">
            <section className="bn-mod bn-code"><h3>SOURCE <small>{ws.component?.name ?? "—"} · {ws.framework} · {ws.attempts} draft{ws.attempts === 1 ? "" : "s"}</small>
              <span className={`bn-state ${ws.phase}`}>{ws.phase === "idle" ? "READY" : ws.phase === "done" ? "PASS" : ws.phase === "failed" ? "FAIL" : ws.phase === "queued" ? "QUEUED" : "RUN " + fmtSecs(ws.elapsed)}</span></h3>
              {ws.runErr && <div className="nx-err">{ws.runErr}</div>}
              {ws.code ? <CodeView ws={ws} range={range} setRange={setRange} /> : <div className="nx-empty">{busy ? "waiting for the Coder…" : "select a target and press ⏎"}</div>}
            </section>
            <section className="bn-mod bn-side">
              <nav className="bn-tabs">{(["events", "ask", "tools"] as Tab[]).map(t => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t.toUpperCase()}</button>)}</nav>
              {tab === "events" && <EventLog ws={ws} />}
              {tab === "ask" && <ChatPane ws={ws} range={range} />}
              {tab === "tools" && <ToolsPane ws={ws} />}
            </section>
          </div>

          <section className="bn-mod bn-scope"><h3>SCOPE <small>tensor shapes measured by forward hooks in the sandbox · segment width ∝ log₂(dim) · <b className="chg">magenta</b> = changed vs input</small></h3>
            {!layers.length && !modelT && <div className="nx-empty">{ws.phase === "done" ? "no layers were hooked (Linear / Conv / LayerNorm / MHA)" : "no signal: shapes appear when the sandbox run passes"}</div>}
            {modelT && <div className="bn-sc model"><span>MODEL</span><Ribbon dims={nums(Object.values(modelT.input ?? {})[0])} /><i>→</i><Ribbon dims={nums(modelT.output)} ref0={nums(Object.values(modelT.input ?? {})[0])} /></div>}
            <div className="bn-sc-grid">{layers.map(([k, v]) => { const a = nums(v.input ?? v.input_shape), b = nums(v.output ?? v.output_shape); return <div className="bn-sc" key={k}><span>{k}</span><Ribbon dims={a} /><i>→</i><Ribbon dims={b} ref0={a} /></div>; })}</div>
          </section>
        </div>

        <aside className="bn-right">
          <section className="bn-mod bn-radarbox"><h3>GRAPH <small>{ws.graph ? `${ws.graph.nodes.length} nodes · ${ws.litIds.size} hot` : "—"}</small></h3><Radar ws={ws} />
            <div className="bn-insp">{focus ? <><small>{focus.type} · {focus.id}</small><b>{focus.name}</b><p>{focus.description}</p></> : <p className="bn-dim">hover or click a blip. green = used by this run's code</p>}</div>
          </section>
          {ws.phase === "failed" && ws.result && <section className="bn-mod bn-fail"><h3>FAULT</h3><p>No pass after {ws.result.attempt_count} attempt(s). {ws.result.error_details}</p></section>}
        </aside>
      </div>
      <footer className="bn-foot"><span>{ws.events.length ? `${ws.events.length} events` : "idle"}</span><span>{ws.events.length ? describeEvent(ws.events[ws.events.length - 1]) : ""}</span><span className="bn-grow" /><span>/ ask · f framework · ⏎ run</span></footer>
    </div>
  );
}
