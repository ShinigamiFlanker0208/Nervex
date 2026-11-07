import { useEffect, useRef, useState } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { ForceSim } from "../../lib/forceSim";
import { sampleText } from "./inkText";
import { IngestPanel } from "../../ui/shared";
import "./hero.css";

const N = 2600;
const INK = [27, 22, 18], RED = [179, 38, 30];
const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (t: number) => t * t * (3 - 2 * t);
const AGENTS = ["Researcher", "Architect", "Coder", "Sandbox", "Critic"];

/**
 * One particle system, four shapes (your law 1): the word -> the paper's real graph -> the agent loop -> dust.
 * Invisible until touched (law 2): dust is almost transparent and only ink-spreads near the cursor.
 * The graph shape uses the REAL ingested graph; with no paper it says so instead of drawing a fake one.
 */
export function Hero({ ws, onEnter }: { ws: Workspace; onEnter: () => void }) {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const live = useRef(ws); live.current = ws;
  const [scene, setScene] = useState(0);
  const [adding, setAdding] = useState(false);
  const S = useRef({ p: 0, mouse: { x: -999, y: -999 }, ready: false, hov: -1 }).current;
  const paper = ws.papers.find(p => p.paper_id === ws.paperId);

  useEffect(() => {
    const canvas = cv.current!, ctx = canvas.getContext("2d")!;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, raf = 0, dead = false; const dpr = Math.min(2, devicePixelRatio || 1);
    const x = new Float32Array(N), y = new Float32Array(N), vx = new Float32Array(N), vy = new Float32Array(N), heat = new Float32Array(N);
    const T = [0, 1, 2, 3].map(() => new Float32Array(N * 2));          // target positions per scene
    const A = [0, 1, 2, 3].map(() => new Float32Array(N));              // target alpha per scene
    const R = [0, 1, 2, 3].map(() => new Float32Array(N));              // target radius per scene
    let graphIdx: { id: string; name: string; type: string; deg: number }[] = [];
    let edges: [number, number][] = [];
    let rnd = (() => { let s = 99; return () => ((s = (s * 16807) % 2147483647) / 2147483647); })();

    const layout = async () => {
      const r = canvas.getBoundingClientRect(); W = r.width; H = r.height; canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      rnd = (() => { let s = 99; return () => ((s = (s * 16807) % 2147483647) / 2147483647); })();
      // scene 0: the word
      const word = await sampleText("Nervex", W, H * .62, N, Math.min(W * .27, 330));
      for (let i = 0; i < N; i++) { T[0][i * 2] = word[i * 2]; T[0][i * 2 + 1] = word[i * 2 + 1] + H * .1; A[0][i] = .92; R[0][i] = 1.35; }
      // scene 1: the real graph (first K particles), the rest are quiet dust
      const g = live.current.graph;
      const K = g ? Math.min(g.nodes.length, 150) : 0;
      graphIdx = []; edges = [];
      const pad = { l: W * .06, r: W * .06, t: H * .2, b: H * .2 };
      if (g && K) {
        // graph lives on the right: the copy owns the left third on wide screens
        const wide = W > 900, gw = wide ? W * .42 : W * .84, gh = wide ? H * .56 : H * .4, gcx = wide ? W * .66 : W * .5, gcy = wide ? H * .52 : H * .68;
        const sim = new ForceSim(g.nodes.slice(0, K), g.edges, gw, gh, 5); sim.padRight = 20;
        for (let i = 0; i < 420; i++) sim.tick();
        const maxDeg = Math.max(1, ...g.nodes.map(n => n.degree));
        sim.nodes.forEach((n, i) => {
          // the sim pins loosely-connected nodes to its walls; pull them in and scatter them so no ruler-straight rows appear
          const jx = (rnd() - .5) * 26, jy = (rnd() - .5) * 26;
          const ex = (n.x - sim.w / 2) * .9 + jx, ey = (n.y - sim.h / 2) * .86 + jy;
          T[1][i * 2] = gcx + ex * (wide ? 1.25 : 1.05); T[1][i * 2 + 1] = gcy + ey * 1.1 + 14; A[1][i] = .95; R[1][i] = 2.2 + 6 * Math.sqrt(n.degree / maxDeg);
          graphIdx.push({ id: n.id, name: n.name ?? n.id, type: n.type ?? "", deg: n.degree });
        });
        const at = new Map(sim.nodes.map((n, i) => [n.id, i])); g.edges.forEach(e => { const a = at.get(e.source), b = at.get(e.target); if (a != null && b != null) edges.push([a, b]); });
      }
      for (let i = K; i < N; i++) { T[1][i * 2] = pad.l + rnd() * (W - pad.l - pad.r); T[1][i * 2 + 1] = pad.t + rnd() * (H - pad.t - pad.b); A[1][i] = .13; R[1][i] = 1.1; }
      // scene 2: five ganglia and the Critic->Coder loop
      const cx = (k: number) => W * (.16 + k * .17), cy = H * .7;
      for (let i = 0; i < N; i++) {
        const k = i % 5, loop = i % 9 === 0;
        if (loop) { const t = rnd(), ax = cx(4), bx = cx(2); T[2][i * 2] = ax + (bx - ax) * t; T[2][i * 2 + 1] = cy - 20 - Math.sin(t * Math.PI) * H * .17; A[2][i] = .55; R[2][i] = 1.2; }
        else { const a = rnd() * 6.283, rr = Math.sqrt(rnd()) * (k === 3 ? 44 : 34); T[2][i * 2] = cx(k) + Math.cos(a) * rr; T[2][i * 2 + 1] = cy + Math.sin(a) * rr; A[2][i] = .85; R[2][i] = 1.5; }
      }
      // scene 3: dust again, fainter; leaves the page quiet for the call to action
      for (let i = 0; i < N; i++) { T[3][i * 2] = rnd() * W; T[3][i * 2 + 1] = rnd() * H; A[3][i] = .09 + rnd() * .1; R[3][i] = 1 + rnd() * .8; }
      if (!S.ready) { for (let i = 0; i < N; i++) { x[i] = rnd() * W; y[i] = rnd() * H; } S.ready = true; }
    };
    layout();
    const ro = new ResizeObserver(() => layout()); ro.observe(canvas);
    let lastGraph = live.current.graph?.paper_id; const relayout = setInterval(() => { const id = live.current.graph?.paper_id; if (id !== lastGraph) { lastGraph = id; layout(); } }, 700);

    const onScroll = () => {
      const r = wrap.current!.getBoundingClientRect(); const span = r.height - innerHeight;
      S.p = clamp(-r.top / Math.max(1, span)); const sc = Math.min(3, Math.floor(S.p * 4 - 0.0001 + 0.08)); setScene(prev => (prev === sc ? prev : sc));
    };
    onScroll(); addEventListener("scroll", onScroll, { passive: true });
    const move = (e: PointerEvent) => { const r = canvas.getBoundingClientRect(); S.mouse.x = e.clientX - r.left; S.mouse.y = e.clientY - r.top; };
    addEventListener("pointermove", move);

    const col = (t: number) => `rgb(${INK[0] + (RED[0] - INK[0]) * t | 0},${INK[1] + (RED[1] - INK[1]) * t | 0},${INK[2] + (RED[2] - INK[2]) * t | 0})`;
    let a0 = 0, a1 = 0, f = 0;
    const frame = (now: number) => {
      if (dead) return; raf = requestAnimationFrame(frame);
      const pos = S.p * 3, s0 = Math.min(2, Math.floor(pos)), fr = smooth(clamp(pos - s0)); a0 = s0; a1 = s0 + 1; f = fr;
      ctx.clearRect(0, 0, W, H);
      const mx = S.mouse.x, my = S.mouse.y, reach = 110;
      // edges (only while the graph shape is on screen)
      const ge = a0 === 0 ? f : a0 === 1 ? 1 - f : 0;
      if (ge > .02 && edges.length) {
        ctx.lineWidth = .7;
        for (const [a, b] of edges) { const near = Math.hypot(x[a] - mx, y[a] - my) < 140 || Math.hypot(x[b] - mx, y[b] - my) < 140;
          ctx.strokeStyle = near ? `rgba(179,38,30,${.5 * ge})` : `rgba(27,22,18,${.17 * ge})`; ctx.beginPath(); ctx.moveTo(x[a], y[a]); ctx.lineTo(x[b], y[b]); ctx.stroke(); }
      }
      let hov = -1, hd = 22 * 22;
      for (let i = 0; i < N; i++) {
        const tx = T[a0][i * 2] + (T[a1][i * 2] - T[a0][i * 2]) * f, ty = T[a0][i * 2 + 1] + (T[a1][i * 2 + 1] - T[a0][i * 2 + 1]) * f;
        const al = A[a0][i] + (A[a1][i] - A[a0][i]) * f, rad = R[a0][i] + (R[a1][i] - R[a0][i]) * f;
        const k = reduce ? 1 : 0.06;
        vx[i] += (tx - x[i]) * k; vy[i] += (ty - y[i]) * k; vx[i] *= reduce ? 0 : .84; vy[i] *= reduce ? 0 : .84;
        const dx = x[i] - mx, dy = y[i] - my, d2 = dx * dx + dy * dy;
        let h = 0;
        if (d2 < reach * reach) { const d = Math.sqrt(d2) || 1, push = (1 - d / reach); if (!reduce) { vx[i] += dx / d * push * 2.2; vy[i] += dy / d * push * 2.2; } h = push; }
        heat[i] += (h - heat[i]) * .18;
        if (reduce) { x[i] = tx; y[i] = ty; } else { x[i] += vx[i]; y[i] += vy[i]; }
        if (i < graphIdx.length && d2 < hd && a0 === 0 && f > .5) { hd = d2; hov = i; }
        // dust is nearly invisible until touched
        const A2 = Math.min(1, al + heat[i] * (al < .3 ? .75 : .25));
        ctx.fillStyle = col(heat[i] > .02 || (i < graphIdx.length && ws.litIds.has(graphIdx[i]?.id)) ? Math.max(heat[i], i < graphIdx.length && ws.litIds.has(graphIdx[i].id) ? .9 : 0) : 0);
        ctx.globalAlpha = A2; ctx.beginPath(); ctx.arc(x[i], y[i], rad * (1 + heat[i] * .5), 0, 6.2832); ctx.fill();
      }
      ctx.globalAlpha = 1;
      // red-pencil circle + name on the node under the cursor (graph scene)
      if (hov >= 0 && ge > .6) {
        const g = graphIdx[hov]; ctx.strokeStyle = "#b3262e"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.ellipse(x[hov] + Math.sin(now / 500) * .5, y[hov], 11 + R[1][hov], 9 + R[1][hov], -.15, 0, 6.2832); ctx.stroke();
        ctx.font = "italic 17px 'Instrument Serif', serif"; ctx.fillStyle = "#1b1612"; ctx.fillText(g.name, Math.min(W - 220, x[hov] + 18), y[hov] - 12);
        ctx.font = "600 9px 'JetBrains Mono', monospace"; ctx.fillStyle = "#b3262e"; ctx.fillText(g.type.toUpperCase(), Math.min(W - 220, x[hov] + 18), y[hov] - 28);
      }
      // labels under the five ganglia
      if (a0 === 1 || a0 === 2) { const vis = a0 === 1 ? f : 1 - f * (a1 === 3 ? 1 : 0); if (vis > .05) { ctx.globalAlpha = vis; ctx.font = "600 10px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
        AGENTS.forEach((n, k) => { ctx.fillStyle = k === 4 ? "#b3262e" : "#1b1612"; ctx.fillText(n.toUpperCase(), W * (.16 + k * .17), H * .7 + 74); }); ctx.textAlign = "left"; ctx.globalAlpha = 1; } }
    };
    raf = requestAnimationFrame(frame);
    return () => { dead = true; cancelAnimationFrame(raf); clearInterval(relayout); ro.disconnect(); removeEventListener("scroll", onScroll); removeEventListener("pointermove", move); };
  }, []);                                                                  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="hero" ref={wrap}>
      <div className="hero-stick">
        <canvas ref={cv} className="hero-canvas" aria-hidden />
        <div className="hero-top"><span className="hero-mark">Nervex</span><span className="hero-folio">a reading machine for research papers</span>
          <button className="hero-skip" onClick={onEnter}>skip the tour →</button></div>

        <section className={`hero-copy s0${scene === 0 ? " on" : ""}`}>
          <p className="hero-kicker">Papers hide machines.</p>
          <p className="hero-scroll">scroll <i /></p>
        </section>
        <section className={`hero-copy s1${scene === 1 ? " on" : ""}`}>
          <p className="hero-kicker">01 · it reads</p>
          <h2>The paper becomes a graph.</h2>
          <p>Every concept, equation and hyperparameter is a node.{paper ? <> This is <b>{paper.title}</b>: <b>{paper.nodes}</b> entities, <b>{paper.edges}</b> relations, each one pulled out of the text by the model you choose. Move through it.</> : <> You haven't fed it a paper yet, so there is nothing honest to draw here. Ingest one at the end of this page.</>}</p>
        </section>
        <section className={`hero-copy s2${scene === 2 ? " on" : ""}`}>
          <p className="hero-kicker">02 · it builds, then it proves</p>
          <h2>Four agents write the code. A sandbox runs it.</h2>
          <p>No network, no privileges, read-only files. If it breaks, the <em>Critic</em> reads the real traceback and the <em>Coder</em> tries again, up to five drafts. What you see afterwards is measured, not guessed.</p>
        </section>
        <section className={`hero-copy s3${scene === 3 ? " on" : ""}`}>
          <p className="hero-kicker">03 · it shows its work</p>
          <h2>Every line cites the paper.<br />Every shape was measured.</h2>
          <div className="hero-cta">
            <button className="hero-btn" onClick={onEnter} disabled={ws.offline}>Enter the reading room</button>
            <button className="hero-link" onClick={() => setAdding(a => !a)}>{ws.papers.length ? "add another paper" : "feed it a paper first"}</button>
          </div>
          {ws.offline && <p className="hero-warn">The API is asleep. <code>uv run python main.py</code></p>}
          {adding && <div className="hero-ingest"><IngestPanel ws={ws} /></div>}
        </section>
        <nav className="hero-dots" aria-hidden>{[0, 1, 2, 3].map(i => <i key={i} className={scene === i ? "on" : ""} />)}</nav>
      </div>
    </div>
  );
}
