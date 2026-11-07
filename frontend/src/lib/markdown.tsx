// Minimal, safe Markdown + KaTeX: paragraphs, bullets, **bold**, `code`, $inline$ and $$block$$ math.
// No dangerouslySetInnerHTML except KaTeX's own output (throwOnError:false, trust:false).
import katex from "katex";
import "katex/dist/katex.min.css";
import { Fragment, ReactNode } from "react";

function math(src: string, display: boolean, key: number) {
  let html = "";
  try { html = katex.renderToString(src, { displayMode: display, throwOnError: false, trust: false }); }
  catch { return <code key={key}>{src}</code>; }
  return <span key={key} dangerouslySetInnerHTML={{ __html: html }} />;
}

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\$\$[^$]+\$\$|\$[^$\n]+\$|\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m: RegExpExecArray | null, k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("$$")) out.push(math(t.slice(2, -2), true, k++));
    else if (t.startsWith("$")) out.push(math(t.slice(1, -1), false, k++));
    else if (t.startsWith("**")) out.push(<strong key={k++}>{t.slice(2, -2)}</strong>);
    else out.push(<code key={k++}>{t.slice(1, -1)}</code>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const blocks = text.replace(/\r/g, "").split(/\n{2,}/);
  return <div className="md">{blocks.map((b, i) => {
    const lines = b.split("\n");
    if (lines.every(l => /^\s*[-*]\s+/.test(l))) return <ul key={i}>{lines.map((l, j) => <li key={j}>{inline(l.replace(/^\s*[-*]\s+/, ""))}</li>)}</ul>;
    const h = /^(#{1,4})\s+(.*)/.exec(b);
    if (h) return <p key={i} className="md-h">{inline(h[2])}</p>;
    return <p key={i}>{lines.map((l, j) => <Fragment key={j}>{j > 0 && <br />}{inline(l)}</Fragment>)}</p>;
  })}</div>;
}
