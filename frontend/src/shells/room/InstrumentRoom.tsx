import { useEffect, useMemo, useRef, useState } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { ForceSim } from "../../lib/forceSim";
import { AGENTS, describeEvent, fmtSecs } from "../../ui/shared";
import "./instrument.css";

const nums = (s: unknown): number[] => (typeof s === "string" ? (s.match(/\d+/g) ?? []).map(Number) : Array.isArray(s) ? (s as number[]).filter(n => typeof n === "number") : []);

/** A tensor shape as an engineering dimension drawing. Box width ~ log(dim); a dimension that changed gets a red leader. */
function Dim({ dims, ref0, label }: { dims: number[]; ref0?: number[]; label: string }) {
  const W = 56, gap = 6, h = 30;
  const ws = dims.map(d => 22 + Math.log2(d + 1) * 5.4);
  const total = ws.reduce((a, b) => a + b, 0) + gap * (dims.length - 1);
  let x = 0;
  return (
    <svg className="ir-dim" viewBox={`0 0 ${Math.max(total, W)} 54`} width={Math.max(total, W)} height="54" role="img" aria-label={`${label} ${dims.join(" by ")}`}>
      {dims.map((d, i) => { const w = ws[i], X = x; x += w + gap; const chg = ref0 && ref0[i] !== d;
        return <g key={i}>
          <rect x={X} y="16" width={w} height={h} fill="none" stroke={chg ? "var(--sig)" : "var(--bone)"} strokeWidth={chg ? 1.4 : .8} />
          <text x={X + w / 2} y="36" textAnchor="middle" className={chg ? "chg" : ""}>{d}</text>
          <line x1={X} y1="9" x2={X + w} y2="9" stroke={chg ? "var(--sig)" : "var(--dimline)"} strokeWidth=".7" markerStart="url(#ir-a)" markerEnd="url(#ir-a)" />
          {chg && <line x1={X + w / 2} y1="46" x2={X + w / 2} y2="53" stroke="var(--sig)" strokeWidth="1" />}
        </g>; })}
    </svg>
  );
}

function Chalk({ ws }: { ws: Workspace }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const live = useRef(ws); live.current = ws;
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!; const dpr = Math.min(2, devicePixelRatio || 1);
    let sim: ForceSim | null = null, key = "", raf = 0, hov: string | null = null;
    const size = () => { const r = cv.getBoundingClientRect(); cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
    size(); const ro = new ResizeObserver(size); ro.observe(cv);
    const wob = (s: string) => { let h = 7; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return ((h % 100) / 100 - .5); };
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const w = cv.width / dpr, h = cv.height / dpr, W = live.current;
      const k = W.graph ? W.graph.paper_id + W.graph.nodes.length : "";
      if (k !== key) { key = k; sim = W.graph ? new ForceSim(W.graph.nodes, W.graph.edges, w, h, 21).adopt(sim) : null; if (sim) sim.padRight = 16; }
      if (sim && Math.abs(sim.w - w) > 3) sim.resize(w, h);
      sim?.tick(); sim?.tick(); ctx.clearRect(0, 0, w, h); if (!sim) return;
      for (const e of sim.edges) {
        const on = W.litIds.has(e.a.id) && W.litIds.has(e.b.id), mx = (e.a.x + e.b.x) / 2 + wob(e.a.id + e.b.id) * 7, my = (e.a.y + e.b.y) / 2 + wob(e.b.id) * 7;
        ctx.strokeStyle = on ? "rgba(226,82,58,.8)" : "rgba(233,225,208,.2)"; ctx.lineWidth = on ? 1.2 : .6;
        ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.quadraticCurveTo(mx, my, e.b.x, e.b.y); ctx.stroke();
      }
      for (const n of sim.nodes) {
        const lit = W.litIds.has(n.id), hot = n.id === (W.hover ?? hov) || n.id === W.selected, r = Math.max(1.6, n.r * .45);
        ctx.fillStyle = lit ? "#e2523a" : "rgba(233,225,208,.75)"; ctx.beginPath(); ctx.arc(n.x + wob(n.id) , n.y + wob(n.id + "y"), r, 0, 6.2832); ctx.fill();
        if (hot || (lit && W.litIds.size < 14)) { ctx.font = "italic 14px 'Instrument Serif', serif"; ctx.fillStyle = "#e9e1d0"; ctx.fillText(n.name ?? n.id, Math.min(w - 120, n.x + 9), n.y - 6); }
      }
    };
    raf = requestAnimationFrame(frame);
    const pick = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); return sim?.nearest(e.clientX - r.left, e.clientY - r.top, 10)?.id ?? null; };
    const move = (e: PointerEvent) => { const id = pick(e); if (id !== hov) { hov = id; live.current.setHover(id); } };
    cv.addEventListener("pointermove", move); cv.addEventListener("click", ((e: PointerEvent) => live.current.setSelected(pick(e))) as any);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); cv.removeEventListener("pointermove", move); };
  }, []);
  return <canvas ref={ref} className="ir-chalk" aria-label="Knowledge graph, chalk on blackboard" />;
}

export function InstrumentRoom({ ws, open, from, onClose }: { ws: Workspace; open: boolean; from: { x: number; y: number }; onClose: () => void }) {
  const [tip, setTip] = useState<number | null>(null);
  useEffect(() => { if (!open) return; const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); }; addEventListener("keydown", h); return () => removeEventListener("keydown", h); }, [open, onClose]);

  const lanes = useMemo(() => AGENTS.map(a => ({ ...a, segs: [] as { i: number; t0: number; t1: number; bad: boolean }[] })), []);
  const timeline = useMemo(() => {
    const L = AGENTS.map(a => ({ key: a.key, label: a.label, segs: [] as { i: number; t0: number; t1: number; bad: boolean }[] }));
    ws.events.forEach((e, i) => { const l = L.find(x => x.key === e.node); if (!l) return; const prev = ws.events[i - 1]?.t ?? 0; l.segs.push({ i, t0: prev, t1: e.t, bad: e.node === "sandbox_executor" && e.status !== "SUCCESS" }); });
    return L;
  }, [ws.events]);                                                      // eslint-disable-line react-hooks/exhaustive-deps
  void lanes;
  const total = Math.max(10, ws.events[ws.events.length - 1]?.t ?? 10);
  const tick = total > 120 ? 30 : total > 50 ? 15 : 5;
  const model = (ws.traces as any).__model__;
  const layers = Object.entries(ws.traces).filter(([k]) => k !== "__model__") as [string, any][];
  const inShape = nums(Object.values(model?.input ?? {})[0]);
  const focus = ws.nodeById(ws.hover ?? ws.selected);
  const date = new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }).toUpperCase();
  const paper = ws.papers.find(p => p.paper_id === ws.paperId);
  const drafts = ws.events.filter(e => e.node === "coder").length;
  const ok = ws.phase === "done";
  const sx = (t: number) => (t / total) * 100;
  const hovEv = tip != null ? ws.events[tip] : null;

  return (
    <div className={`ir${open ? " open" : ""}`} style={{ ["--ox" as any]: `${from.x}px`, ["--oy" as any]: `${from.y}px` }} aria-hidden={!open} role="dialog" aria-label="Instrument room">
      <svg width="0" height="0" style={{ position: "absolute" }}><defs><marker id="ir-a" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 3 L6 0 M0 3 L6 6" stroke="#8c8473" strokeWidth=".8" fill="none" /></marker></defs></svg>
      <header className="ir-head">
        <div><small>PLATE II</small><h2>The instrument room</h2></div>
        <div className="ir-head-r"><span className={`ir-stamp ${ok ? "ok" : "bad"}`}>{ok ? "VERIFIED BY SANDBOX" : ws.phase === "failed" ? "NOT VERIFIED" : "RUNNING"}</span>
          <button onClick={onClose}>back to the page <kbd>esc</kbd></button></div>
      </header>

      <div className="ir-grid">
        <section className="ir-fig a">
          <h3><b>Fig. A</b> The run, as measured <span>each bar is the real time between two agent events</span></h3>
          <div className="ir-time">
            {timeline.map(l => (
              <div className="ir-lane" key={l.key}><em>{l.label}</em>
                <div className="ir-track">
                  {l.segs.map(s => <i key={s.i} className={`ir-seg${s.bad ? " bad" : ""}${l.key === "critic" ? " crit" : ""}`} style={{ left: `${sx(s.t0)}%`, width: `${Math.max(.8, sx(s.t1) - sx(s.t0))}%` }}
                    onMouseEnter={() => setTip(s.i)} onMouseLeave={() => setTip(null)} />)}
                </div>
              </div>))}
            <div className="ir-axis">{Array.from({ length: Math.floor(total / tick) + 1 }, (_, k) => k * tick).map(t => <span key={t} style={{ left: `${sx(t)}%` }}>{fmtSecs(t)}</span>)}</div>
          </div>
          <p className="ir-note">{hovEv ? <><b>+{hovEv.t.toFixed(1)}s</b> {describeEvent(hovEv)}</> : ws.events.length ? <>{drafts} draft{drafts === 1 ? "" : "s"}, {fmtSecs(total)} end to end. Hover a bar.</> : "No run yet."}</p>
        </section>

        <section className="ir-fig b">
          <h3><b>Fig. B</b> Tensor shapes <span>measured by forward hooks inside the sandbox · red = a dimension changed</span></h3>
          {!model && !layers.length && <p className="ir-empty">{ok ? "No layers were hooked (Linear, Conv, LayerNorm, MultiheadAttention)." : "Shapes appear when a draft runs cleanly."}</p>}
          {model && <div className="ir-row model"><span>model</span><Dim dims={inShape} label="input" /><i>→</i><Dim dims={nums(model.output)} ref0={inShape} label="output" /></div>}
          <div className="ir-layers">{layers.map(([k, v]) => { const a = nums(v.input ?? v.input_shape), b = nums(v.output ?? v.output_shape); return <div className="ir-row" key={k}><span title={k}>{k}</span><Dim dims={a} label={k + " in"} /><i>→</i><Dim dims={b} ref0={a} label={k + " out"} /></div>; })}</div>
        </section>

        <section className="ir-fig c">
          <h3><b>Fig. C</b> What the code is built from <span>{ws.litIds.size} node{ws.litIds.size === 1 ? "" : "s"} cited</span></h3>
          <Chalk ws={ws} />
          <div className="ir-insp">{focus ? <><small>{focus.type} · {focus.id}</small><strong>{focus.name}</strong><p>{focus.description}</p></> : <p className="ir-empty">Red marks the nodes the generated code cites through <code># Implements:</code>. Hover one.</p>}</div>
        </section>
      </div>

      <footer className="ir-title">
        <div><small>paper</small><b>{paper?.title ?? "—"}</b></div>
        <div><small>component</small><b>{ws.component?.name ?? "—"}</b></div>
        <div><small>framework</small><b>{ws.framework}</b></div>
        <div><small>drafts</small><b>{drafts}</b></div>
        <div><small>layers traced</small><b>{layers.length}</b></div>
        <div><small>model</small><b>{ws.health?.llm.provider ?? "—"}</b></div>
        <div><small>date</small><b>{date}</b></div>
        <div><small>checked by</small><b>docker · no network · uid 1000</b></div>
      </footer>
    </div>
  );
}
