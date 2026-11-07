import { useMemo, useState } from "react";
import type { Workspace } from "../../../data/useWorkspace";
import { tagForLine } from "../../../lib/codeTags";

/**
 * B · The Dissection Table. The generated model, laid out from what the sandbox MEASURED: every hooked
 * layer in execution order, grouped by its module path, repeated blocks folded ("× 6"). Click a block to
 * see the code that defines it, the equation it cites, and its shapes. "Lift it off" runs an ablation.
 */
interface Leaf { path: string; leaf: string; kind: string; inp: number[]; out: number[] }
interface Group { key: string; label: string; reps: number; leaves: Leaf[] }

const nums = (s: unknown): number[] => (typeof s === "string" ? (s.match(/\d+/g) ?? []).map(Number) : Array.isArray(s) ? (s as any[]).filter(n => typeof n === "number") : []);
const kindOf = (leaf: string) => /norm|ln/i.test(leaf) ? "LayerNorm" : /conv/i.test(leaf) ? "Conv" : /attn|attention|mha/i.test(leaf) ? "Attention" : /emb/i.test(leaf) ? "Embedding" : "Linear";

function build(traces: Record<string, any>): Group[] {
  const leaves: Leaf[] = Object.entries(traces).filter(([k]) => k !== "__model__").map(([path, v]) => {
    const leaf = path.split(".").pop() ?? path;
    return { path, leaf, kind: kindOf(leaf), inp: nums(v?.input ?? v?.input_shape), out: nums(v?.output ?? v?.output_shape) };
  });
  // group key: the path with the last numeric index collapsed ("encoder.layers.3.ffn.fc1" -> "encoder.layers.#")
  const groups: Group[] = [];
  for (const l of leaves) {
    const m = /^(.*?\.)(\d+)\./.exec(l.path);
    const key = m ? m[1] + "#" : l.path.split(".").slice(0, -1).join(".") || "(root)";
    const idx = m ? +m[2] : 0;
    let g = groups[groups.length - 1];
    if (!g || g.key !== key) { g = { key, label: key.replace(/\.#$/, "").replace(/\./g, " › ") || "model", reps: 0, leaves: [] }; groups.push(g); }
    g.reps = Math.max(g.reps, idx + 1);
    if (idx === 0) g.leaves.push(l);                              // draw one instance of a repeated block
  }
  return groups;
}

export function TableView({ ws }: { ws: Workspace }) {
  const [sel, setSel] = useState<Leaf | null>(null);
  const [lifting, setLifting] = useState<string | null>(null);
  const groups = useMemo(() => build(ws.traces), [ws.traces]);
  const model = (ws.traces as any).__model__;
  const lines = useMemo(() => ws.code.split("\n"), [ws.code]);
  const busy = ws.phase === "running" || ws.phase === "queued" || ws.ablation.busy;

  if (!groups.length && !model) return (
    <div className="tv-empty"><div className="tv-mat" /><p>{ws.phase === "idle" ? "Nothing on the table. Build a component and its measured layers will be laid out here." : ws.phase === "done" ? "The run passed but no layers were hooked (Linear, Conv, LayerNorm, MultiheadAttention)." : "The table fills when a draft runs cleanly in the sandbox."}</p></div>);

  // where in the code is this layer defined / used, and which equation tags govern those lines
  const where = sel ? lines.map((l, i) => ({ n: i + 1, l })).filter(x => new RegExp(`\\bself\\.${sel.leaf}\\b`).test(x.l)) : [];
  const cites = sel ? [...new Set(where.map(x => tagForLine(ws.tags, lines, x.n)).filter(Boolean) as string[])] : [];
  const lift = (l: Leaf) => { setLifting(l.path); ws.runAblation(`Remove the layer \`${l.path}\` (${l.kind}) from the model. Keep every remaining tensor shape valid so the forward pass still runs.`); };
  const maxW = Math.max(1, ...groups.flatMap(g => g.leaves.map(l => l.out[l.out.length - 1] ?? 1)));

  return (
    <div className="tv">
      <div className="tv-mat">
        <div className="tv-ruler">{Array.from({ length: 24 }, (_, i) => <i key={i} />)}</div>
        {model && <div className="tv-io in"><small>input</small><b>{Object.entries(model.input ?? {}).map(([k, v]) => `${k} [${nums(v).join(", ")}]`).join("  ·  ")}</b></div>}
        {groups.map(g => (
          <section key={g.key} className={`tv-group${g.reps > 1 ? " rep" : ""}`}>
            <header><span>{g.label}</span>{g.reps > 1 && <em>× {g.reps}</em>}</header>
            <div className="tv-blocks">
              {g.leaves.map(l => {
                const w = 62 + 36 * Math.log2(1 + (l.out[l.out.length - 1] ?? 1)) / Math.log2(1 + maxW);
                const changed = l.inp.length && l.out.length && l.inp.join() !== l.out.join();
                return <button key={l.path} className={`tv-b k-${l.kind}${sel?.path === l.path ? " on" : ""}${lifting === l.path && ws.ablation.busy ? " lifting" : ""}`} style={{ width: `${w}%` }} onClick={() => setSel(l)}>
                  <b>{l.leaf}</b><i>{l.kind}</i><span className={changed ? "chg" : ""}>[{l.inp.join(", ")}] → [{l.out.join(", ")}]</span></button>;
              })}
            </div>
          </section>))}
        {model && <div className="tv-io out"><small>output</small><b>[{nums(model.output).join(", ")}]</b></div>}
      </div>

      <aside className="tv-insp">
        {!sel && <p className="mg-dim">Click a block on the table.<br />Width ∝ log of its output size. Red shapes changed inside the block.</p>}
        {sel && <>
          <small>{sel.kind} · measured</small><h4>{sel.leaf}</h4><code className="tv-path">{sel.path}</code>
          <dl><dt>in</dt><dd>[{sel.inp.join(", ")}]</dd><dt>out</dt><dd>[{sel.out.join(", ")}]</dd></dl>
          <h5>In the code</h5>
          {where.length ? <pre>{where.slice(0, 6).map(x => `${String(x.n).padStart(3)}  ${x.l.trim()}`).join("\n")}</pre> : <p className="mg-dim">Not found by name (it may be built inside a loop or helper).</p>}
          <h5>From the paper</h5>
          {cites.length ? cites.map(c => <p key={c} className="tv-cite" onMouseEnter={() => ws.setHover(c)} onMouseLeave={() => ws.setHover(null)}>{ws.nodeById(c)?.name ?? c}<span>{ws.nodeById(c)?.description ?? ""}</span></p>) : <p className="mg-dim">No <code># Implements:</code> tag covers these lines.</p>}
          <button className="tv-lift" disabled={busy || !ws.code} onClick={() => lift(sel)}>{ws.ablation.busy && lifting === sel.path ? "lifting it off, re-running…" : "Lift it off the table"}</button>
          <p className="mg-dim tv-note">Rewrites the module without this layer, runs it in the sandbox, and files the result in the notebook.</p>
          {ws.ablation.err && lifting === sel.path && <div className="nx-err">{ws.ablation.err}</div>}
        </>}
      </aside>
    </div>
  );
}
