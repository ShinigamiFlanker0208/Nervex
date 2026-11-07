import { ReactNode, useMemo, useState } from "react";
import { api } from "../../../api/client";
import type { RunStatus } from "../../../api/types";
import type { Workspace } from "../../../data/useWorkspace";
import { Markdown } from "../../../lib/markdown";
import { fmtSecs } from "../../../ui/shared";

/**
 * A · The Lab Notebook. Every run and every ablation is a dated entry, newest first: the hypothesis
 * (what was built, from how many equations), the drafts, the verdict, the measurement. The run in progress
 * is the open page at the top. Nothing is ever torn out: failed runs stay.
 */
const day = (t: number) => new Date(t * 1000).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long" });
const hm = (t: number) => new Date(t * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });

type Entry = { kind: "run"; t: number; id: string; comp: string; fw: string; ok: boolean; attempts: number; dur: number; layers: number; lines: number }
  | { kind: "abl"; t: number; id: string; comp: string; prompt: string; before: any; after: any; expl: string };

function Detail({ id }: { id: string }) {
  const [r, setR] = useState<RunStatus | null | "err">(null);
  useMemo(() => { api.historyItem(id).then(setR).catch(() => setR("err")); }, [id]);
  if (r === null) return <p className="mg-dim">opening the entry…</p>;
  if (r === "err") return <p className="mg-dim">This entry could not be read.</p>;
  const res = r.events.find(e => e.node === "researcher");
  const drafts = r.events.filter(e => e.node === "coder");
  return (
    <div className="nb-detail">
      <p><b>Given:</b> {res?.formulas ?? 0} equation{res?.formulas === 1 ? "" : "s"} from the graph{res?.hyperparameters?.length ? `, hyperparameters ${res.hyperparameters.slice(0, 6).join(", ")}` : ""}.</p>
      <ol>{r.events.filter(e => e.node === "sandbox_executor" || e.node === "critic").map(e => (
        <li key={e.i} className={e.node === "critic" ? "crit" : e.status === "SUCCESS" ? "ok" : "bad"}>
          <time>+{e.t.toFixed(0)}s</time>{e.node === "critic" ? <span className="nb-pencil">{(e.feedback ?? "").replace(/\s+/g, " ").slice(0, 260)}</span> : <span>sandbox: {e.status === "SUCCESS" ? `passed, ${e.layers?.length ?? 0} layers traced` : e.status}</span>}</li>))}</ol>
      <p className="mg-dim">{drafts.length} draft{drafts.length === 1 ? "" : "s"} · final module {(r.result?.generated_code ?? "").split("\n").length} lines</p>
    </div>
  );
}

export function NotebookView({ ws, live, onReopen }: { ws: Workspace; live: ReactNode; onReopen: (id: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const [only, setOnly] = useState(false);
  const name = (id: string) => ws.components.find(c => c.id === id)?.name ?? ws.nodeById(id)?.name ?? id;
  const entries = useMemo<Entry[]>(() => {
    const e: Entry[] = [
      ...ws.shelf.map(r => ({ kind: "run" as const, t: r.created_at ?? 0, id: r.run_id, comp: r.component, fw: r.framework, ok: r.status === "SUCCESS", attempts: r.attempts, dur: r.duration, layers: r.layers, lines: r.lines })),
      ...ws.ablations.map(a => ({ kind: "abl" as const, t: a.created_at, id: a.ablation_id, comp: a.component, prompt: a.mutation_prompt, before: a.baseline_metrics, after: a.ablated_metrics, expl: a.diff_explanation })),
    ];
    return e.filter(x => !only || x.comp === ws.component?.id).sort((a, b) => b.t - a.t);
  }, [ws.shelf, ws.ablations, only, ws.component]);
  const days: [string, Entry[]][] = [];
  entries.forEach(e => { const d = day(e.t); const last = days[days.length - 1]; if (last && last[0] === d) last[1].push(e); else days.push([d, [e]]); });
  let no = entries.length;
  const layers = (m: any) => m?.traced_layers ?? m?.layers?.length ?? "–";

  return (
    <div className="nb">
      <div className="nb-head"><h3>Notebook</h3><span>{ws.shelf.length} run{ws.shelf.length === 1 ? "" : "s"} · {ws.ablations.length} experiment{ws.ablations.length === 1 ? "" : "s"}</span>
        {ws.component && <label><input type="checkbox" checked={only} onChange={e => setOnly(e.target.checked)} /> only {ws.component.name}</label>}</div>
      {ws.phase !== "idle" && <article className="nb-entry live"><div className="nb-margin"><b>today</b><i>{ws.phase === "done" || ws.phase === "failed" ? "just now" : "in progress"}</i></div><div className="nb-body">{live}</div></article>}
      {!entries.length && ws.phase === "idle" && <p className="mg-dim nb-blank">A blank page. Every run and experiment will be written here, including the ones that fail.</p>}
      {days.map(([d, es]) => (
        <section key={d}><h4 className="nb-day">{d}</h4>
          {es.map(e => { const n = no--; return e.kind === "run" ? (
            <article key={e.id} className={`nb-entry ${e.ok ? "ok" : "bad"}`}>
              <div className="nb-margin"><b>No. {n}</b><i>{hm(e.t)}</i></div>
              <div className="nb-body">
                <h5>Built <em>{name(e.comp)}</em> in {e.fw}</h5>
                <p>{e.attempts} draft{e.attempts === 1 ? "" : "s"}, {fmtSecs(e.dur)}{e.ok ? `, ${e.layers} layers measured, ${e.lines} lines.` : "."} <span className={`nb-verdict ${e.ok ? "ok" : "bad"}`}>{e.ok ? "verified" : "not verified"}</span></p>
                <div className="nb-acts"><button onClick={() => setOpen(o => o === e.id ? null : e.id)}>{open === e.id ? "close" : "read the entry"}</button><button onClick={() => onReopen(e.id)}>reopen on the table</button></div>
                {open === e.id && <Detail id={e.id} />}
              </div></article>
          ) : (
            <article key={e.id} className="nb-entry abl">
              <div className="nb-margin"><b>No. {n}</b><i>{hm(e.t)}</i></div>
              <div className="nb-body">
                <h5>Experiment on <em>{name(e.comp)}</em></h5>
                <blockquote>{e.prompt}</blockquote>
                <div className="nb-ba"><div><small>before</small><b>{layers(e.before)}</b><span>layers</span></div><i>→</i><div><small>after</small><b>{layers(e.after)}</b><span>layers</span></div></div>
                <div className="nb-expl"><Markdown text={e.expl.slice(0, 700)} /></div>
              </div></article>); })}
        </section>))}
    </div>
  );
}
