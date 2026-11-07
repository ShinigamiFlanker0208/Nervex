import { useEffect, useRef, useState } from "react";

// Switch the LLM provider at runtime. The server tests the choice with one tiny call and only switches if it answers.
interface Prov { id: string; default_model: string; has_key: boolean }
const LABEL: Record<string, string> = { nvidia: "NVIDIA NIM", google: "Google Gemini", openai: "OpenAI", openai_compatible: "OpenAI-compatible", ollama: "Ollama (local)" };

export function ProviderSwitch({ busy, onChanged }: { busy: boolean; onChanged?: () => void }) {
  const [list, setList] = useState<Prov[]>([]);
  const [active, setActive] = useState(""); const [model, setModel] = useState("");
  const [open, setOpen] = useState(false);
  const [pick, setPick] = useState<string | null>(null);           // provider needing a key / url form
  const [key, setKey] = useState(""); const [url, setUrl] = useState(""); const [mdl, setMdl] = useState("");
  const [state, setState] = useState<{ testing?: string; msg?: string; ok?: boolean }>({});
  const box = useRef<HTMLDivElement>(null);
  const load = () => fetch("/api/v1/system/providers").then(r => r.json()).then(d => { setList(d.providers); setActive(d.active); setModel(d.model); }).catch(() => {});
  useEffect(() => { load(); }, []);
  useEffect(() => { const h = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); }; addEventListener("mousedown", h); return () => removeEventListener("mousedown", h); }, []);

  const test = async (id: string, extra: Record<string, string> = {}) => {
    setState({ testing: id });
    try {
      const r = await fetch("/api/v1/system/provider", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: id, ...extra }) });
      const d = await r.json();
      if (!r.ok) { setState({ ok: false, msg: d.detail ?? "refused" }); return; }
      if (d.ok) { setState({ ok: true, msg: `${LABEL[id]} answered in ${d.latency_s}s` }); setPick(null); setKey(""); await load(); onChanged?.(); setTimeout(() => setOpen(false), 900); }
      else setState({ ok: false, msg: d.error === "upstream_auth" ? "key rejected" : d.error === "connection" ? "could not reach it" : d.error === "rate_limited" ? "rate limited, key is valid but busy" : d.error === "upstream_error" || d.error === "timeout" ? "provider busy or slow right now, try again" : `${d.error}: ${d.detail?.slice(0, 120)}` });
    } catch { setState({ ok: false, msg: "API unreachable" }); }
  };
  const choose = (p: Prov) => {
    if (p.id === active) return;
    setState({}); setMdl("");
    if (p.has_key && p.id !== "openai_compatible") test(p.id); else setPick(p.id);
  };

  return (
    <div className="pvs" ref={box}>
      <button className="pvs-btn" disabled={busy} title={busy ? "finish the run first" : "change the model provider"} onClick={() => setOpen(o => !o)}>
        <i className="pvs-dot" />{LABEL[active] ?? active ?? "…"}<small>{model.split("/").pop()}</small><b>▾</b></button>
      {open && <div className="pvs-menu" role="menu">
        <p className="pvs-h">Model provider <span>switches every agent, no restart</span></p>
        {list.map(p => (
          <button key={p.id} role="menuitem" className={`pvs-item${p.id === active ? " on" : ""}`} disabled={!!state.testing} onClick={() => choose(p)}>
            <span>{LABEL[p.id] ?? p.id}</span><small>{p.default_model.split("/").pop()}</small>
            <em className={p.has_key ? "k" : ""}>{p.id === active ? "active" : state.testing === p.id ? "testing…" : (p.id === "ollama" || p.id === "openai_compatible") ? "no key needed" : p.has_key ? "key found" : "needs key"}</em></button>))}
        {pick && <form className="pvs-form" onSubmit={e => { e.preventDefault(); test(pick, { ...(key ? { api_key: key } : {}), ...(url ? { base_url: url } : {}), ...(mdl ? { model: mdl } : {}) }); }}>
          {pick !== "ollama" && <input type="password" autoComplete="off" placeholder={pick === "openai_compatible" ? "API key (if the server needs one)" : `${LABEL[pick]} API key`} value={key} onChange={e => setKey(e.target.value)} />}
          {(pick === "openai_compatible" || pick === "ollama") && <input placeholder={pick === "ollama" ? "URL (default http://localhost:11434/v1)" : "Base URL, e.g. http://localhost:8001/v1"} value={url} onChange={e => setUrl(e.target.value)} />}
          <input placeholder="model (optional, else the default)" value={mdl} onChange={e => setMdl(e.target.value)} />
          <button disabled={!!state.testing || (pick === "openai_compatible" && !url) || (["nvidia", "google", "openai"].includes(pick) && !key)}>{state.testing ? "testing…" : "Test & switch"}</button>
          <p>The key stays in the server's memory for this session only. To keep it, put it in <code>.env</code>.</p>
        </form>}
        {state.msg && <p className={`pvs-msg ${state.ok ? "ok" : "bad"}`}>{state.ok ? "✓ " : "✗ "}{state.msg}</p>}
      </div>}
    </div>
  );
}
