import { useEffect, useState } from "react";
import { useWorkspace } from "./data/useWorkspace";
import { SynapseShell } from "./shells/synapse/SynapseShell";
import { MarginaliaShell } from "./shells/marginalia/MarginaliaShell";
import { BenchShell } from "./shells/bench/BenchShell";

export const FACES = [
  { id: "synapse", label: "Synapse" },
  { id: "marginalia", label: "Marginalia" },
  { id: "bench", label: "Bench" },
] as const;
type Face = (typeof FACES)[number]["id"];

const read = (): Face => {
  const f = new URLSearchParams(location.search).get("ui");
  return (FACES.find(x => x.id === f)?.id ?? "synapse") as Face;
};

export function App() {
  const ws = useWorkspace();                 // ONE state for all faces: switching faces never loses a run
  const [face, setFace] = useState<Face>(read);
  useEffect(() => { const h = () => setFace(read()); addEventListener("popstate", h); return () => removeEventListener("popstate", h); }, []);
  const go = (f: Face) => (e: React.MouseEvent) => { e.preventDefault(); history.pushState(null, "", `?ui=${f}`); setFace(f); };
  const faces = <nav className="faces">{FACES.map(f => <a key={f.id} href={`?ui=${f.id}`} className={face === f.id ? "on" : ""} onClick={go(f.id)}>{f.label}</a>)}</nav>;
  if (face === "marginalia") return <MarginaliaShell ws={ws} faces={faces} />;
  if (face === "bench") return <BenchShell ws={ws} faces={faces} />;
  return <SynapseShell ws={ws} faces={faces} />;
}
