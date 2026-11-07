import { useCallback, useEffect, useRef, useState } from "react";
import type { Workspace } from "../../data/useWorkspace";
import { ReadingRoom } from "./ReadingRoom";
import { Hero } from "./Hero";
import { HowItWorks } from "./HowItWorks";
import { InstrumentRoom } from "./InstrumentRoom";

const KEY = "nervex.entered";
const seen = () => { try { return sessionStorage.getItem(KEY) === "1"; } catch { return false; } };

/** Hero -> reading room (paper) -> instrument room (blackboard). The gate is a room change (law 4). */
export function RoomShell({ ws, faces }: { ws: Workspace; faces: React.ReactNode }) {
  const [entered, setEntered] = useState(seen);
  const [open, setOpen] = useState(false);
  const [from, setFrom] = useState({ x: 900, y: 120 });
  const wasRunning = useRef(false);

  const enter = useCallback(() => { try { sessionStorage.setItem(KEY, "1"); } catch { /* private mode */ } setEntered(true); scrollTo(0, 0); }, []);
  const tour = useCallback(() => { try { sessionStorage.removeItem(KEY); } catch { /* ignore */ } setEntered(false); setTimeout(() => document.getElementById("how")?.scrollIntoView(), 60); }, []);
  const openFrom = useCallback((el?: Element | null) => {
    const r = (el ?? document.querySelector(".mg-stamp") ?? document.querySelector(".mg-gate"))?.getBoundingClientRect();
    setFrom(r ? { x: r.left + r.width / 2, y: r.top + r.height / 2 } : { x: innerWidth * .7, y: 140 }); setOpen(true);
  }, []);

  // the moment a run passes the sandbox, the room changes by itself, once per run
  useEffect(() => {
    const running = ws.phase === "running" || ws.phase === "queued";
    if (wasRunning.current && ws.phase === "done") setTimeout(() => openFrom(), 700);
    wasRunning.current = running;
  }, [ws.phase, openFrom]);

  if (!entered) return <><Hero ws={ws} onEnter={enter} /><HowItWorks ws={ws} onEnter={enter} /></>;
  return <>
    <ReadingRoom ws={ws} faces={faces} onOpenInstrument={() => openFrom(document.querySelector(".mg-gate"))} onTour={tour} />
    <InstrumentRoom ws={ws} open={open} from={from} onClose={() => setOpen(false)} />
  </>;
}
