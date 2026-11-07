import { useEffect, useRef } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { ForceSim } from "../../lib/forceSim";

/** Graph as a radar: a sweep line lights nodes as it passes; nodes the run touched stay hot (lime). */
export function Radar({ ws }: { ws: Workspace }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const live = useRef(ws); live.current = ws;
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!; const dpr = Math.min(2, devicePixelRatio || 1);
    let sim: ForceSim | null = null, key = "", raf = 0, hov: string | null = null;
    const swept = new Map<string, number>();
    const size = () => { const r = cv.getBoundingClientRect(); cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
    size(); const ro = new ResizeObserver(size); ro.observe(cv);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const w = cv.width / dpr, h = cv.height / dpr, ws = live.current, cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 6;
      const k = ws.graph ? ws.graph.paper_id + ws.graph.nodes.length : "";
      if (k !== key) { key = k; sim = ws.graph ? new ForceSim(ws.graph.nodes, ws.graph.edges, w, h, 3).adopt(sim) : null; if (sim) sim.padRight = 30; }
      if (sim && Math.abs(sim.w - w) > 3) sim.resize(w, h);
      sim?.tick(); sim?.tick();
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(62,242,224,.16)"; ctx.lineWidth = 1;
      for (const f of [1, .66, .33]) { ctx.beginPath(); ctx.arc(cx, cy, R * f, 0, 6.2832); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
      const ang = reduce ? 0 : (now / 3200) % 6.2832;
      if (!reduce) { const g = ctx.createConicGradient(ang - 1.2, cx, cy); g.addColorStop(0, "rgba(62,242,224,0)"); g.addColorStop(.19, "rgba(62,242,224,.22)"); g.addColorStop(.2, "rgba(62,242,224,0)"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 6.2832); ctx.fill(); }
      if (!sim) return;
      const clipR = R - 4;
      const P = (n: { x: number; y: number }) => { // squash the sim rectangle into the circle
        const dx = (n.x - w / 2) / (w / 2), dy = (n.y - h / 2) / (h / 2), m = Math.max(1, Math.hypot(dx, dy)); return { x: cx + dx / m * clipR * Math.min(1, Math.hypot(dx, dy)), y: cy + dy / m * clipR * Math.min(1, Math.hypot(dx, dy)) };
      };
      ctx.lineWidth = .6;
      for (const e of sim.edges) { const A = P(e.a), B = P(e.b), on = ws.litIds.has(e.a.id) && ws.litIds.has(e.b.id); ctx.strokeStyle = on ? "rgba(166,255,77,.5)" : "rgba(62,242,224,.1)"; ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke(); }
      for (const n of sim.nodes) {
        const p = P(n); const a = (Math.atan2(p.y - cy, p.x - cx) + 6.2832) % 6.2832;
        if (!reduce && Math.abs(((a - ang + 9.4248) % 6.2832) - 3.1416) < .06) swept.set(n.id, now);
        const sw = swept.get(n.id), glow = sw ? Math.max(0, 1 - (now - sw) / 1800) : reduce ? .4 : 0;
        const lit = ws.litIds.has(n.id), hot = n.id === (ws.hover ?? hov) || n.id === ws.selected;
        ctx.globalAlpha = Math.min(1, .22 + glow * .8 + (lit ? .6 : 0) + (hot ? .6 : 0));
        ctx.fillStyle = lit ? "#a6ff4d" : n.type === "ArchitectureComponent" ? "#3ef2e0" : n.type === "Formula" ? "#ffb84d" : "#7d8ea3";
        ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(1.5, n.r * .38) + (hot ? 2 : 0), 0, 6.2832); ctx.fill(); ctx.globalAlpha = 1;
        if (hot) { ctx.strokeStyle = "#ff3ea5"; ctx.strokeRect(p.x - 6, p.y - 6, 12, 12); }
      }
      (cv as any)._P = P;
    };
    raf = requestAnimationFrame(frame);
    const pick = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, P = (cv as any)._P; if (!sim || !P) return null;
      let best: string | null = null, bd = 100; for (const n of sim.nodes) { const p = P(n), d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = n.id; } } return best;
    };
    const move = (e: PointerEvent) => { const id = pick(e); if (id !== hov) { hov = id; live.current.setHover(id); } };
    const click = (e: PointerEvent) => live.current.setSelected(pick(e));
    cv.addEventListener("pointermove", move); cv.addEventListener("click", click as any); cv.addEventListener("pointerleave", () => { hov = null; live.current.setHover(null); });
    return () => { cancelAnimationFrame(raf); ro.disconnect(); cv.removeEventListener("pointermove", move); cv.removeEventListener("click", click as any); };
  }, []);
  return <canvas ref={ref} className="bn-radar" aria-label="Graph radar" />;
}
