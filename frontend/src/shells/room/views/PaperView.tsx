import { Fragment, ReactNode, useMemo } from "react";
import katex from "katex";
import type { Component } from "../../../api/types";
import type { Workspace } from "../../../data/useWorkspace";

/**
 * C · The Living Paper. The paper's own text (the chunks Nervex stored), with every buildable component
 * underlined where the prose mentions it. Clicking a mention builds it. The build appears as a pull-out
 * in the margin next to the first sentence that names it.
 */
const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function PaperView({ ws, onOpenTable }: { ws: Workspace; onOpenTable: () => void }) {
  const busy = ws.phase === "running" || ws.phase === "queued";
  // names worth matching: >= 4 chars, longest first so "Multi-Head Attention" wins over "Attention"
  const byName = useMemo(() => {
    const m = new Map<string, Component>();
    for (const c of ws.components) { const n = (c.name ?? "").trim(); if (n.length >= 4 && !m.has(n.toLowerCase())) m.set(n.toLowerCase(), c); }
    return m;
  }, [ws.components]);
  const re = useMemo(() => {
    const names = [...byName.keys()].sort((a, b) => b.length - a.length).map(esc);
    return names.length ? new RegExp(`\\b(${names.join("|")})\\b`, "gi") : null;
  }, [byName]);

  const firstMention = useMemo(() => {   // passage index where the current component is first named
    if (!ws.component || !ws.passages) return -1;
    const n = (ws.component.name ?? "").toLowerCase();
    return ws.passages.findIndex(p => p.text.toLowerCase().includes(n));
  }, [ws.component, ws.passages]);

  // Clean what OCR + chunking left behind: "[\[13\]](#page-10-0)" -> "[13]", escaped "\*", <sup>, and the
  // 150-char overlap the chunker repeats at the start of each chunk (drop any prefix seen at the previous tail).
  const clean = useMemo(() => {
    if (!ws.passages) return null;
    const fix = (t: string) => t.replace(/\[\\?\[([^\]]+?)\\?\]\]\(#[^)]*\)/g, "[$1]").replace(/\[([^\]]+)\]\(#page[^)]*\)/g, "$1")
      .replace(/\\([*_\[\]#])/g, "$1").replace(/<\/?su[bp]>/g, "").replace(/\s+\[(\d+(?:,\s*\d+)*)\]/g, "\u2009[$1]");
    let prevTail = "";
    return ws.passages.map(p => {
      let t = fix(p.text);
      for (let k = Math.min(400, t.length, prevTail.length); k >= 40; k--) if (prevTail.endsWith(t.slice(0, k))) { t = t.slice(k).replace(/^\S*\s/, ""); break; }
      prevTail = fix(p.text).slice(-420);
      return { ...p, text: t };
    });
  }, [ws.passages]);

  if (ws.passages === null) return <p className="mg-dim">Opening the paper…</p>;
  if (!ws.passages.length) return <p className="mg-dim">No stored text for this paper. Ingest it again to rebuild its passages.</p>;

  const mark = (text: string, key: string): ReactNode[] => {
    // equations were vaulted at ingest; show the token as a small mark instead of raw <MATH_BLOCK_0012>
    const parts = text.split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|<MATH_BLOCK_\d+>)/g);
    return parts.flatMap((part, pi) => {
      if (/^<MATH_BLOCK_\d+>$/.test(part)) return [<span key={`${key}m${pi}`} className="pv-eq" title={part}>equation</span>];
      if (/^\$[\s\S]*\$$/.test(part)) {
        const disp = part.startsWith("$$"), src = part.replace(/^\$\$?|\$\$?$/g, "");
        const html = katex.renderToString(src, { throwOnError: false, displayMode: disp, trust: false });
        return [<span key={`${key}k${pi}`} className={disp ? "pv-disp" : "pv-tex"} dangerouslySetInnerHTML={{ __html: html }} />];
      }
      if (!re) return [part];
      const out: ReactNode[] = []; let last = 0; re.lastIndex = 0; let m: RegExpExecArray | null;
      while ((m = re.exec(part))) {
        const c = byName.get(m[0].toLowerCase()); if (!c) continue;
        if (m.index > last) out.push(part.slice(last, m.index));
        const done = ws.shelf.some(r => r.component === c.id && r.status === "SUCCESS");
        out.push(<button key={`${key}-${pi}-${m.index}`} className={`pv-mark${ws.component?.id === c.id ? " cur" : ""}${done ? " done" : ""}${ws.hover === c.id ? " hot" : ""}`}
          disabled={busy} title={`Build ${c.name} (${c.formulas} formula${c.formulas === 1 ? "" : "s"} nearby)`}
          onMouseEnter={() => ws.setHover(c.id)} onMouseLeave={() => ws.setHover(null)} onClick={() => ws.startRun(c)}>{m[0]}</button>);
        last = m.index + m[0].length;
      }
      out.push(part.slice(last)); return out;
    });
  };

  const block = (t: string, key: string) => {
    const h = /^(#{1,4})\s+(.*)$/.exec(t.trim());
    if (h) return <h3 key={key} className={`pv-h h${h[1].length}`}>{mark(h[2].replace(/\*\*/g, ""), key)}</h3>;
    if (/^\s*\|/.test(t)) {
      const all = t.trim().split("\n"), cut = all.findIndex(l => !/^\s*\|/.test(l));
      if (cut > 0) return <Fragment key={key}>{block(all.slice(0, cut).join("\n"), key + "t")}{block(all.slice(cut).join("\n"), key + "r")}</Fragment>;
      const rows = all.filter(r => !/^\s*\|?[\s:|-]+\|?\s*$/.test(r)).map(r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
      return <div key={key} className="pv-tablewrap"><table className="pv-tbl"><tbody>{rows.map((r, ri) => <tr key={ri}>{r.map((c, ci) => ri === 0 ? <th key={ci}>{mark(c.replace(/<\/?b>|\*\*/g, ""), `${key}${ri}${ci}`)}</th> : <td key={ci}>{mark(c.replace(/<\/?b>|\*\*/g, ""), `${key}${ri}${ci}`)}</td>)}</tr>)}</tbody></table></div>;
    }
    return <p key={key}>{mark(t.replace(/\*\*([^*]+)\*\*/g, "$1").replace(/<\/?su[bp]>/g, ""), key)}</p>;
  };

  return (
    <div className="pv">
      <p className="pv-hint"><span className="pv-mark demo">underlined</span> phrases are components Nervex can build. Click one. <i>{ws.passages.length} passages, as the OCR read them.</i></p>
      {(clean ?? []).map((p, i) => (
        <section key={p.id} className="pv-pass">
          {p.text.split(/\n{2,}/).map((t, j) => <Fragment key={j}>{block(t, `${i}-${j}`)}</Fragment>)}
          {i === firstMention && ws.phase !== "idle" && (
            <aside className={`pv-pull ${ws.phase}`}>
              <small>{ws.phase === "done" ? "built & verified" : ws.phase === "failed" ? "built, not verified" : "being built"}</small>
              <b>{ws.component?.name}</b>
              <pre>{(ws.code || "…").split("\n").filter(l => l.trim()).slice(0, 9).join("\n")}</pre>
              {ws.phase === "done" && <button onClick={onOpenTable}>lay it on the table →</button>}
            </aside>)}
        </section>))}
    </div>
  );
}
