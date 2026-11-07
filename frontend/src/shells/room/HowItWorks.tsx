import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { RunStatus, SystemConfig } from "../../api/types";
import type { Workspace } from "../../data/useWorkspace";
import { describeEvent, fmtSecs, NODE_LABEL } from "../../ui/shared";
import "./how.css";

/** Pipeline as an engineering figure. Every label is a real module in the repo. */
function PipelineFigure({ active }: { active: string | null }) {
  const ing = [["PDF", "upload, magic-byte check"], ["OCR", "Marker / Surya, own process"], ["Mask", "LaTeX → vault tokens"], ["Chunk", "semantic, ~1k chars"], ["Extract", "LLM → typed entities"]];
  const loop = [["researcher", "Researcher"], ["architect", "Architect"], ["coder", "Coder"], ["sandbox_executor", "Sandbox"], ["critic", "Critic"]];
  return (
    <svg className="hw-fig" viewBox="0 0 1000 360" role="img" aria-label="Nervex pipeline: ingestion into graph and vectors, then the agent loop">
      <defs><marker id="hw-ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="currentColor" strokeWidth="1.2" /></marker></defs>
      <text x="0" y="16" className="hw-cap">I · INGESTION</text>
      {ing.map(([t, s], i) => <g key={t} transform={`translate(${i * 200},30)`}>
        <rect width="160" height="58" rx="2" /><text x="12" y="26" className="hw-t">{t}</text><text x="12" y="45" className="hw-s">{s}</text>
        {i < 4 && <line x1="162" y1="29" x2="196" y2="29" markerEnd="url(#hw-ar)" />}</g>)}
      <line x1="880" y1="90" x2="880" y2="130" markerEnd="url(#hw-ar)" />
      <g transform="translate(560,134)"><rect width="200" height="50" className="store" /><text x="14" y="22" className="hw-t">Neo4j</text><text x="14" y="40" className="hw-s">graph, keyed by (node, paper)</text></g>
      <g transform="translate(780,134)"><rect width="200" height="50" className="store" /><text x="14" y="22" className="hw-t">ChromaDB</text><text x="14" y="40" className="hw-s">passages, paper-scoped</text></g>
      <path d="M660 186 C660 214 80 206 80 246" markerEnd="url(#hw-ar)" className="dash" />
      <text x="0" y="216" className="hw-cap">II · THE LOOP</text>
      {loop.map(([k, t], i) => <g key={k} transform={`translate(${i * 200},250)`} className={active === k ? "on" : ""}>
        <rect width="160" height="54" rx="2" /><text x="12" y="33" className="hw-t">{t}</text>
        {i < 3 && <line x1="162" y1="27" x2="196" y2="27" markerEnd="url(#hw-ar)" />}</g>)}
      <path d="M720 304 C720 330 700 340 680 340 L 900 340" className="none" />
      <path d="M760 277 L 796 277" markerEnd="url(#hw-ar)" className="red" />
      <text x="770" y="268" className="hw-s red">fail</text>
      <path d="M880 306 C880 352 560 352 560 310" markerEnd="url(#hw-ar)" className="red" />
      <text x="690" y="352" className="hw-s red">traceback → fix (≤ attempts)</text>
      <text x="610" y="244" className="hw-s good">pass → code + measured shapes</text>
    </svg>
  );
}

function Replay({ ws }: { ws: Workspace }) {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [k, setK] = useState(0);
  const pick = ws.shelf.find(r => r.status === "SUCCESS" && r.attempts > 1) ?? ws.shelf[0];
  useEffect(() => { if (pick) api.historyItem(pick.run_id).then(r => { setRun(r); setK(r.events.length - 1); }).catch(() => {}); }, [pick?.run_id]);   // eslint-disable-line react-hooks/exhaustive-deps
  if (!pick) return <p className="hw-dim">No saved run yet. Generate any component once and it will appear here, frame by frame.</p>;
  if (!run) return <p className="hw-dim">loading the run…</p>;
  const ev = run.events[k];
  const code = [...run.events.slice(0, k + 1)].reverse().find(e => e.code)?.code ?? "";
  return (
    <div className="hw-replay">
      <div className="hw-rp-head"><b>{ws.nodeById(run.meta.target_component)?.name ?? ws.components.find(c => c.id === run.meta.target_component)?.name ?? run.meta.target_component}</b> · {run.meta.target_framework} · {run.result?.execution_status} after {run.result?.attempt_count} draft(s) · <i>a real run from your shelf</i></div>
      <input type="range" min={0} max={run.events.length - 1} value={k} onChange={e => setK(+e.target.value)} aria-label="Scrub through the run" />
      <div className="hw-rp-ticks">{run.events.map((e, i) => <button key={i} className={`${i === k ? "on " : ""}${e.node}${e.node === "sandbox_executor" && e.status !== "SUCCESS" ? " bad" : ""}`} onClick={() => setK(i)} title={NODE_LABEL[e.node]} />)}</div>
      <div className="hw-rp-body">
        <div><small>+{ev.t.toFixed(1)}s</small><h4>{NODE_LABEL[ev.node]}</h4><p>{describeEvent(ev)}</p>
          {ev.node === "critic" && ev.feedback && <blockquote>{ev.feedback.slice(0, 420)}</blockquote>}
          {ev.node === "sandbox_executor" && ev.status !== "SUCCESS" && ev.log_tail && <pre>{ev.log_tail.slice(-420)}</pre>}</div>
        <pre className="hw-rp-code">{code ? code.split("\n").slice(0, 26).join("\n") + (code.split("\n").length > 26 ? "\n…" : "") : "(no code yet at this point)"}</pre>
      </div>
    </div>
  );
}

export function HowItWorks({ ws, onEnter }: { ws: Workspace; onEnter: () => void }) {
  const [cfg, setCfg] = useState<SystemConfig | null>(null);
  const [hot, setHot] = useState<string | null>(null);
  useEffect(() => { api.config().then(setCfg).catch(() => {}); }, []);
  const v = (x: unknown, fallback = "…") => (x === undefined || x === null ? fallback : String(x));
  const paper = ws.papers.find(p => p.paper_id === ws.paperId);
  const stats = useMemo(() => {
    const done = ws.shelf.filter(r => r.status === "SUCCESS");
    return { n: ws.shelf.length, ok: done.length, first: done.filter(r => r.attempts === 1).length, avg: done.length ? done.reduce((a, r) => a + r.duration, 0) / done.length : 0 };
  }, [ws.shelf]);
  const steps: [string, string, string, JSX.Element][] = [
    ["researcher", "§ 4", "The Researcher reads before anyone writes.", <>It resolves the component you picked to real graph ids (slug, then name search, scoped to this paper), pulls their two-hop neighbourhood from Neo4j and the closest passages from ChromaDB. Every equation is handed on as <code>[node_id] latex</code>, so later the code can cite it.</>],
    ["architect", "§ 5", "The Architect draws the contract.", <>A class skeleton with a shape comment on every line (<code># [B, S, D] → [B, S, D]</code>) and the dummy inputs the sandbox will feed it, with dtypes: masks are boolean, ids are int64. No math yet. A wrong shape here would waste every later attempt, so it is fixed first.</>],
    ["coder", "§ 6", "The Coder fills in the math.", <>It must obey a harness contract: the top-level module builds with no arguments (defaults come from the paper's hyperparameters), it is defined last, and each implemented equation gets a <code># Implements: &lt;Node_ID: …&gt;</code> tag. That tag is the thread you can pull later, from a line of code back to the paper.</>],
    ["sandbox_executor", "§ 7", "The sandbox is the only judge.", <>A fresh container per draft: network <b>{v(cfg?.sandbox.network)}</b>, capabilities <b>{v(cfg?.sandbox.capabilities)}</b>, user <b>{v(cfg?.sandbox.user)}</b>, files <b>{v(cfg?.sandbox.workspace)}</b>, <b>{v(cfg?.sandbox.memory_mb)} MB</b> memory, killed at <b>{v(cfg?.sandbox.timeout_s)} s</b>. Before anything runs, the AST is compiled and every import checked against a whitelist. Forward hooks record the shape at every layer and the result leaves the container on stdout, because <code>/tmp</code> dies with it.</>],
    ["critic", "§ 8", "The Critic reads the real traceback.", <>Not a guess at what might be wrong: the exact exception from the run. It tells the Coder what to change, the Coder writes the next draft, and the loop goes again, at most <b>{v(cfg?.loop.max_attempts)}</b> times. Every rejected draft is kept, so you can see how the code got there.</>],
  ];

  return (
    <article className="hw" id="how">
      <header className="hw-head">
        <p className="hw-kicker">How it works</p>
        <h2>A paper goes in.<br /><em>Verified code comes out.</em><br />Nothing in between is hidden.</h2>
        <p className="hw-lede">Nervex is a pipeline of small, checkable steps. Below is every one of them, with the numbers it really runs with on this machine{cfg ? "" : " (start the API to see them)"}.</p>
        {stats.n > 0 && <div className="hw-ledger">
          <div><b>{stats.n}</b><span>runs on the shelf</span></div><div><b>{stats.ok}</b><span>verified by the sandbox</span></div>
          <div><b>{stats.first}</b><span>passed on the first draft</span></div><div><b>{fmtSecs(stats.avg)}</b><span>average to a pass</span></div>
          {paper && <div><b>{paper.nodes}</b><span>entities in {paper.title}</span></div>}</div>}
      </header>

      <section className="hw-sec"><div className="hw-n">§ 1</div><div>
        <h3>Why this exists</h3>
        <p>A paper describes a model in prose and equations. Turning that into code is where most mistakes happen: a transposed matrix, a missing mask, a dimension that only fails at batch size two. An assistant that writes the code is easy. One that <em>proves it runs</em>, shows the shapes it measured, and points every line back to the sentence it came from, is the point.</p>
      </div></section>

      <section className="hw-sec wide"><div className="hw-n">§ 2</div><div>
        <h3>The whole machine, on one page</h3>
        <PipelineFigure active={hot} />
      </div></section>

      <section className="hw-sec"><div className="hw-n">§ 3</div><div>
        <h3>Reading the paper</h3>
        <ol className="hw-list">
          <li><b>OCR in its own process.</b> Marker and Surya start their own vLLM server. On an 8 GB card that would starve everything else, so OCR runs in a child process capped at <b>{cfg ? Math.round(cfg.ingest.ocr_gpu_fraction * 100) : "…"}%</b> of VRAM and is torn down afterwards. Results are cached by the file's hash.</li>
          <li><b>Equations are vaulted.</b> Every LaTeX block becomes a token like <code>&lt;MATH_BLOCK_0007&gt;</code> before chunking, so no chunk boundary can cut an equation in half. The model sees the original LaTeX again only for the tokens in its chunk.</li>
          <li><b>Entities, typed.</b> Each chunk goes to the LLM with a Pydantic schema: Concept, Formula, ArchitectureComponent, Hyperparameter, and the edges between them. <b>{v(cfg?.ingest.extraction_concurrency)}</b> chunks in parallel; a rate limit pauses every worker, not just the unlucky one.</li>
          <li><b>A quality gate before writing.</b> If more than <b>{cfg ? Math.round(cfg.ingest.max_failed_fraction * 100) : "…"}%</b> of chunks fail, nothing is written. A bad run can never overwrite a good graph. A re-ingest replaces the paper's subgraph in one transaction.</li>
        </ol>
      </div></section>

      {steps.map(([key, n, title, body]) => (
        <section key={key} className={`hw-sec step${hot === key ? " hot" : ""}`} onMouseEnter={() => setHot(key)} onMouseLeave={() => setHot(null)}>
          <div className="hw-n">{n}</div><div><h3>{title}</h3><p>{body}</p></div>
        </section>))}

      <section className="hw-sec wide"><div className="hw-n">§ 9</div><div>
        <h3>Replay a real run</h3>
        <p>Not an animation of how it should work. These are the events a run actually emitted, in order, with the code as it was at that moment. Drag through it.</p>
        <Replay ws={ws} />
      </div></section>

      <section className="hw-sec"><div className="hw-n">§ 10</div><div>
        <h3>After it passes</h3>
        <ul className="hw-list">
          <li><b>Ask the margin.</b> Highlight lines and ask. The explainer follows the <code># Implements:</code> tags into the graph and answers from the paper's own text, citing the nodes it used.</li>
          <li><b>Ablate.</b> Describe a change ("remove the output projection"). The baseline is re-verified first, then the mutated module runs in the same sandbox and the measured layers are compared.</li>
          <li><b>Transpile.</b> PyTorch ⇄ JAX/Equinox, and the result is executed on batched inputs, not just parsed. A pass means it runs and returns finite numbers of the right shape; it does not prove the math is identical.</li>
        </ul>
      </div></section>

      <section className="hw-sec"><div className="hw-n">§ 11</div><div>
        <h3>Bring your own model</h3>
        <p>Every agent asks one factory for its model. Set <code>LLM_PROVIDER</code> to <b>nvidia</b>, <b>google</b>, <b>openai</b>, <b>ollama</b> or any <b>openai_compatible</b> server (vLLM, LM Studio, llama.cpp). This instance is running <b>{v(cfg?.llm.provider)}</b> · <code>{v(cfg?.llm.model)}</code>. The key never leaves your machine except to the provider you chose.</p>
      </div></section>

      <section className="hw-sec"><div className="hw-n">§ 12</div><div>
        <h3>What it does not promise</h3>
        <ul className="hw-list dim">
          <li>A pass proves the code <em>runs</em> with the shapes shown. It does not prove the math matches the paper.</li>
          <li>The graph is only as good as the extraction. Some "formulas" are complexity notes like <code>O(n)</code>.</li>
          <li>Free API tiers rate-limit hard: ingesting one paper can take 10–25 minutes.</li>
          <li>Needs an NVIDIA GPU for the sandbox. The API binds to localhost; it runs code, so do not expose it.</li>
        </ul>
      </div></section>

      <footer className="hw-end"><button className="hero-btn" onClick={onEnter} disabled={ws.offline}>Enter the reading room</button><p>MIT licensed · runs on your machine · <i>Nervex</i></p></footer>
    </article>
  );
}
