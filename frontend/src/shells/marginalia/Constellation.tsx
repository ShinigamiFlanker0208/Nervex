import { useEffect, useRef } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { ForceSim } from "../../lib/forceSim";

/** The paper's graph drawn as a small ink plate: hairlines, dots, red-pencil circles on what the code uses. */
export function Constellation({ ws }: { ws: Workspace }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const live = useRef(ws); live.current = ws;
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!; const dpr = Math.min(2, devicePixelRatio || 1);
    let sim: ForceSim | null = null, key = "", raf = 0, hov: string | null = null;
    const size = () => { const r = cv.getBoundingClientRect(); cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
    size(); const ro = new ResizeObserver(size); ro.observe(cv);
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const w = cv.width / dpr, h = cv.height / dpr, ws = live.current;
      const k = ws.graph ? ws.graph.paper_id + ws.graph.nodes.length : "";
      if (k !== key) { key = k; sim = ws.graph ? new ForceSim(ws.graph.nodes, ws.graph.edges, w, h, 11).adopt(sim) : null; if (sim) sim.padRight = 14; }
      if (sim && (Math.abs(sim.w - w) > 3)) sim.resize(w, h);
      sim?.tick(); sim?.tick();
      ctx.clearRect(0, 0, w, h); if (!sim) return;
      ctx.lineWidth = .6;
      for (const e of sim.edges) { const on = ws.litIds.has(e.a.id) && ws.litIds.has(e.b.id); ctx.strokeStyle = on ? "rgba(179,38,30,.55)" : "rgba(27,22,18,.16)"; ctx.beginPath(); ctx.moveTo(e.a.x, e.a.y); ctx.lineTo(e.b.x, e.b.y); ctx.stroke(); }
      for (const n of sim.nodes) {
        const lit = ws.litIds.has(n.id), hot = n.id === (ws.hover ?? hov) || n.id === ws.selected;
        const r = Math.max(1.4, n.r * .42);
        ctx.fillStyle = lit ? "#b3262e" : "rgba(27,22,18,.78)"; ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 6.2832); ctx.fill();
        if (lit || hot) { ctx.strokeStyle = "#b3262e"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(n.x + Math.sin(now / 900 + n.x) * .4, n.y, r + 3.2, 0, 6.2832); ctx.stroke(); }
        if (hot) { ctx.font = "italic 13px 'Instrument Serif', serif"; ctx.fillStyle = "#1b1612"; ctx.fillText(n.name ?? n.id, Math.min(w - 130, n.x + 9), n.y - 6); }
      }
    };
    raf = requestAnimationFrame(frame);
    const move = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); const n = sim?.nearest(e.clientX - r.left, e.clientY - r.top, 10); const id = n?.id ?? null; if (id !== hov) { hov = id; live.current.setHover(id); } };
    const click = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); live.current.setSelected(sim?.nearest(e.clientX - r.left, e.clientY - r.top, 10)?.id ?? null); };
    cv.addEventListener("pointermove", move); cv.addEventListener("click", click as any); cv.addEventListener("pointerleave", () => { hov = null; live.current.setHover(null); });
    return () => { cancelAnimationFrame(raf); ro.disconnect(); cv.removeEventListener("pointermove", move); cv.removeEventListener("click", click as any); };
  }, []);
  return <canvas ref={ref} className="mg-plate-canvas" aria-label="Knowledge graph plate" />;
}
