import { useEffect, useState } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { AGENTS, ChatPane, CodeView, EventLog, IngestPanel, OfflineNotice, Telemetry, ToolsPane, describeEvent, fmtSecs } from "../../ui/shared";
import { SynapseCanvas } from "./SynapseCanvas";
import "./synapse.css";

type Tab = "code" | "shapes" | "ask" | "tools";

export function SynapseShell({ ws, faces }: { ws: Workspace; faces: React.ReactNode }) {
  const [tab, setTab] = useState<Tab>("code");
  const [range, setRange] = useState<[number, number] | null>(null);
  const [adding, setAdding] = useState(false);
  const hasRun = ws.phase !== "idle";
  const node = ws.nodeById(ws.hover ?? ws.selected);
  const ag = AGENTS.find(a => a.key === ws.activeNode);
  useEffect(() => { if (ws.phase === "done") setTab("shapes"); }, [ws.phase]);
  useEffect(() => { if (ws.phase === "running" || ws.phase === "queued") setTab("code"); }, [ws.phase]);
  const isComponent = (id: string) => ws.components.find(c => c.id === id);
  const empty = !ws.papers.length && !ws.offline;

  return (
    <div className="sx" data-shell="synapse">
      <SynapseCanvas ws={ws} inset={{ left: 300, right: hasRun ? 560 : 20 }} />
      <header className="sx-top glass">
        <div className="sx-brand"><svg width="22" height="22" viewBox="0 0 32 32"><circle cx="16" cy="16" r="4" fill="#a78bfa" /><path d="M16 12V3M20 18l8 5M12 18l-8 5" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" /></svg><span className="serif">nervex</span></div>
        <select value={ws.paperId ?? ""} onChange={e => ws.setPaperId(e.target.value)} disabled={!ws.papers.length} aria-label="Paper">
          {ws.papers.length === 0 && <option>no papers yet</option>}
          {ws.papers.map(p => <option key={p.paper_id} value={p.paper_id}>{p.title}</option>)}
        </select>
        <button className="sx-btn ghost" onClick={() => setAdding(a => !a)}>＋ paper</button>
        <div className="sx-health" title={ws.health ? `LLM: ${ws.health.llm.provider} · ${ws.health.llm.model}` : ""}>
          {ws.health && Object.entries(ws.health.subsystems).map(([k, v]) => <span key={k} className={v ? "ok" : "bad"} title={k}>●<i>{k.replace("_", " ")}</i></span>)}
          {ws.health?.llm.provider && <em>{ws.health.llm.provider}</em>}
        </div>
        {faces}
      </header>

      <OfflineNotice ws={ws} />
      {(empty || adding) && <div className="sx-center glass"><IngestPanel ws={ws} /><button className="sx-link" onClick={() => setAdding(false)} hidden={empty}>close</button></div>}
      {ws.loadErr && !ws.offline && <div className="sx-toast">{ws.loadErr}</div>}

      <aside className="sx-left glass">
        <h3>Targets</h3>
        <div className="sx-seg">{["PyTorch", "JAX"].map(f => <button key={f} className={ws.framework === f ? "on" : ""} onClick={() => ws.setFramework(f)}>{f}</button>)}</div>
        <ul>
          {ws.components.map(c => (
            <li key={c.id} className={(ws.component?.id === c.id ? "cur " : "") + (ws.hover === c.id || ws.selected === c.id ? "hot" : "")}
              onMouseEnter={() => ws.setHover(c.id)} onMouseLeave={() => ws.setHover(null)}>
              <button onClick={() => ws.startRun(c)} disabled={ws.phase === "running" || ws.phase === "queued"} title={c.description ?? ""}>
                <span>{c.name ?? c.id}</span><small title="formulas the code can cite">{c.formulas ? `${c.formulas}ƒ` : c.degree}</small><b>▸</b>
              </button>
            </li>
          ))}
          {!ws.components.length && <li className="sx-dim">{ws.paperId ? "No components found" : "Ingest a paper to begin"}</li>}
        </ul>
        {node && <div className="sx-node"><small>{node.type}</small><strong>{node.name ?? node.id}</strong><p>{node.description}</p>
          {isComponent(node.id) && <button className="sx-btn" disabled={ws.phase === "running" || ws.phase === "queued"} onClick={() => ws.startRun(isComponent(node.id)!)}>Generate this →</button>}</div>}
      </aside>

      {hasRun && (
        <aside className="sx-right glass">
          <div className="sx-tabs">{(["code", "shapes", "ask", "tools"] as Tab[]).map(t => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}
            <span className="sx-grow" /><span className={`sx-state ${ws.phase}`}>{ws.phase}</span></div>
          {ws.runErr && <div className="nx-err">{ws.runErr}</div>}
          {tab === "code" && (ws.code ? <CodeView ws={ws} range={range} setRange={setRange} /> : <EventLog ws={ws} />)}
          {tab === "shapes" && <Telemetry ws={ws} />}
          {tab === "ask" && <ChatPane ws={ws} range={range} />}
          {tab === "tools" && <ToolsPane ws={ws} />}
          {ws.phase === "failed" && ws.result && <div className="nx-err">Did not pass after {ws.result.attempt_count} attempt(s){ws.result.error_details ? `: ${ws.result.error_details}` : ""}. The last code is shown above.</div>}
        </aside>
      )}

      <footer className="sx-foot">
        {hasRun ? <>
          <span className="sx-time">{fmtSecs(ws.elapsed)}</span>
          <span className="sx-say">{ws.phase === "queued" ? "waiting for the GPU" : ws.phase === "running" ? `${ag?.label ?? "…"} · ${ag?.verb ?? ""}` : ws.phase === "done" ? `verified in the sandbox · ${ws.attempts} attempt${ws.attempts === 1 ? "" : "s"}` : "run ended without a passing result"}</span>
          {ws.events.length > 1 && <span className="sx-last">{describeEvent(ws.events[ws.events.length - 1])}</span>}
        </> : <span className="sx-say dim">{ws.graph ? "Pick a target on the left. Hover a node to read it. Click to pin." : ""}</span>}
      </footer>
    </div>
  );
}
