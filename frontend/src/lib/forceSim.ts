// Tiny force layout (no deps). ~150 nodes x O(n^2) repulsion is ~22k pair ops per tick: fine at 60 fps.
import type { GEdge, GNode } from "../api/types";

export interface SimNode extends GNode { x: number; y: number; vx: number; vy: number; r: number; fixed?: boolean }
export interface SimEdge { a: SimNode; b: SimNode; type: string }

export class ForceSim {
  linked = new Set<string>();   // node ids that have at least one edge inside this drawn subset
  nodes: SimNode[]; edges: SimEdge[]; index = new Map<string, SimNode>(); alpha = 1; padRight = 130;   // Synapse draws labels right of nodes; small plates lower this
  constructor(nodes: GNode[], edges: GEdge[], public w: number, public h: number, seed = 7) {
    let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const maxDeg = Math.max(1, ...nodes.map(n => n.degree));
    this.nodes = nodes.map(n => ({
      ...n, x: w / 2 + (rnd() - .5) * w * .6, y: h / 2 + (rnd() - .5) * h * .6, vx: 0, vy: 0,
      r: 2.5 + 9 * Math.sqrt(n.degree / maxDeg),
    }));
    this.nodes.forEach(n => this.index.set(n.id, n));
    this.edges = edges.flatMap(e => {
      const a = this.index.get(e.source), b = this.index.get(e.target);
      return a && b && a !== b ? [{ a, b, type: e.type }] : [];
    });
    this.edges.forEach(e => { this.linked.add(e.a.id); this.linked.add(e.b.id); });
  }
  /** Keep the previous layout for nodes that already existed, so growing the graph doesn't scramble it. */
  adopt(old: ForceSim | null) {
    if (!old) return this;
    for (const n of this.nodes) { const o = old.index.get(n.id); if (o) { n.x = o.x; n.y = o.y; } }
    return this;
  }
  resize(w: number, h: number) { this.w = w; this.h = h; this.alpha = Math.max(this.alpha, .3); }
  kick(a = .5) { this.alpha = Math.max(this.alpha, a); }
  tick() {
    const { nodes, edges, w, h } = this;
    this.alpha = 1;      // constant forces + damping => a steady equilibrium (annealing collapsed loose nodes into a ball)
    const k = Math.sqrt((w * h) / Math.max(1, nodes.length)) * 0.5;
    for (let i = 0; i < nodes.length; i++) {
      const p = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const q = nodes[j]; let dx = p.x - q.x, dy = p.y - q.y; let d2 = dx * dx + dy * dy;
        if (d2 > k * k * 6) continue;
        if (d2 < 1) { dx = Math.random() - .5; dy = Math.random() - .5; d2 = 1; }
        const d = Math.sqrt(d2), f = (k * k) / d / d * 0.35;
        p.vx += dx * f; p.vy += dy * f; q.vx -= dx * f; q.vy -= dy * f;
      }
    }
    for (const e of edges) {
      const dx = e.b.x - e.a.x, dy = e.b.y - e.a.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - k * .9) / d * 0.05;
      e.a.vx += dx * f; e.a.vy += dy * f; e.b.vx -= dx * f; e.b.vy -= dy * f;
    }
    for (const n of nodes) {
      // gravity is NOT scaled by alpha: when the layout cools, loosely-connected nodes must stay near the middle
      // nodes with no edge in this subset have nothing to hold them: stronger gravity makes a halo, not a border
      const gk = this.linked.has(n.id) ? 1 : 3.4;
      n.vx += (w / 2 - n.x) * 0.0018 * gk; n.vy += (h / 2 - n.y) * 0.0028 * gk;
      // soft containment: an elliptical wall that pushes back harder the further out a node is. A hard clamp alone
      // pins every unconnected node to the border and draws ruler-straight rows of dots.
      const ex = (n.x - w / 2) / Math.max(1, (w - this.padRight) / 2 - 10), ey = (n.y - h / 2) / Math.max(1, h / 2 - 24), er = Math.hypot(ex, ey);
      if (er > .78) { const push = (er - .78) * 2.6; n.vx -= (ex / er) * push; n.vy -= (ey / er) * push; }
      const sp = Math.hypot(n.vx, n.vy); if (sp > 7) { n.vx *= 7 / sp; n.vy *= 7 / sp; }
      if (!n.fixed) { n.x += n.vx; n.y += n.vy; }
      n.vx *= 0.78; n.vy *= 0.78;
      n.x = Math.min(w - this.padRight, Math.max(Math.min(40, this.padRight), n.x)); n.y = Math.min(h - 40, Math.max(40, n.y));
    }
    return true;
  }
  nearest(x: number, y: number, maxD = 22): SimNode | null {
    let best: SimNode | null = null, bd = maxD * maxD;
    for (const n of this.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2; if (d < bd + n.r * n.r) { bd = d; best = n; } }
    return best;
  }
}
