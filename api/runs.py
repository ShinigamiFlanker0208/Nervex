"""
Live generation runs: the agent loop reported node-by-node.

Why this exists: POST /generate/code blocks for minutes and says nothing. A UI that shows the agents
working needs REAL events, not a fake spinner. `RunStore` holds an append-only event list per run;
the client polls `GET /generate/runs/{id}?since=N` and receives only the new events.

`summarize` is a pure function (LangGraph node update -> small JSON event) so it is unit-testable and
the UI never has to understand the full state.
"""
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_FORMULA_ID = re.compile(r"^\[([^\]]+)\]")
TAIL = 1800


def summarize(node: str, update: Dict[str, Any]) -> Dict[str, Any]:
    ev: Dict[str, Any] = {"node": node}
    if node == "researcher":
        formulas = update.get("relevant_formulas") or []
        ev["node_ids"] = [m.group(1) for f in formulas if (m := _FORMULA_ID.match(str(f)))]
        ev["formulas_tex"] = [{"id": m.group(1), "latex": str(f)[m.end():].strip()[:400]}
                              for f in formulas[:24] if (m := _FORMULA_ID.match(str(f)))]
        ev["formulas"] = len(formulas)
        ev["hyperparameters"] = list((update.get("hyperparameters") or {}).keys())[:12]
    elif node == "architect":
        ev["expected_inputs"] = update.get("expected_inputs") or {}
        ev["skeleton"] = str(update.get("structural_skeleton") or "")[:TAIL]
    elif node == "coder":
        code = str(update.get("current_code") or "")
        ev["code"] = code
        ev["lines"] = code.count("\n") + 1 if code else 0
        ev["attempt"] = update.get("attempt_count")
    elif node == "sandbox_executor":
        ev["status"] = update.get("execution_status")
        ev["attempt"] = update.get("attempt_count")
        ev["layers"] = [k for k in (update.get("tensor_shape_traces") or {}) if k != "__model__"]
        logs = update.get("execution_logs") or []
        last = logs[-1] if logs else None
        ev["log_tail"] = (str(last.get("traceback") or last) if isinstance(last, dict) else str(last or ""))[-TAIL:]
    elif node == "critic":
        ev["feedback"] = str(update.get("critic_feedback") or "")[:TAIL]
    return ev


class RunStore:
    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, meta: Dict[str, Any]) -> str:
        rid = uuid.uuid4().hex
        with self._lock:
            self._runs[rid] = {"run_id": rid, "state": "queued", "meta": meta, "events": [],
                               "result": None, "created_at": time.time(), "started_at": None}
        return rid

    def update(self, rid: str, **fields: Any) -> None:
        with self._lock:
            self._runs[rid].update(fields)

    def event(self, rid: str, ev: Dict[str, Any]) -> None:
        with self._lock:
            run = self._runs[rid]
            base = run["started_at"] or run["created_at"]
            run["events"].append({"i": len(run["events"]), "t": round(time.time() - base, 2), **ev})

    def get(self, rid: str, since: int = 0) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(rid)
            if run is None:
                return None
            return {**{k: v for k, v in run.items() if k != "events"},
                    "events": list(run["events"][max(0, since):]), "total_events": len(run["events"])}


runs = RunStore()


# ----------------------------------------------------------------------------------------------
# Persistence: finished runs are written to disk so the UI can show a "shelf" of what was verified,
# and a restart does not erase the evidence. One JSON file per run; ids are validated (hex only) so a
# crafted run_id can never walk out of RUNS_DIR.
import json
from pathlib import Path

_ID = re.compile(r"^[0-9a-f]{32}$")


def _dir() -> Path:
    from config import settings
    d = Path(settings.RUNS_DIR); d.mkdir(parents=True, exist_ok=True); return d


def persist(run_id: str) -> Optional[Path]:
    run = runs.get(run_id, 0)
    if run is None or not _ID.match(run_id):
        return None
    path = _dir() / f"{run_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**run, "finished_at": time.time()}))
    tmp.replace(path)                                    # atomic: a crash never leaves a half-written file
    return path


def summary_of(run: Dict[str, Any]) -> Dict[str, Any]:
    res = run.get("result") or {}
    ev = run.get("events") or []
    meta = run.get("meta") or {}
    traces = res.get("tensor_shape_traces") or {}
    return {
        "run_id": run["run_id"], "paper_id": meta.get("paper_id"), "component": meta.get("target_component"),
        "framework": meta.get("target_framework"), "status": res.get("execution_status", "UNKNOWN"),
        "attempts": res.get("attempt_count", 0), "duration": ev[-1]["t"] if ev else 0,
        "layers": len([k for k in traces if k != "__model__"]), "lines": (res.get("generated_code") or "").count("\n") + 1 if res.get("generated_code") else 0,
        "created_at": run.get("created_at"),
    }


def list_history(paper_id: Optional[str] = None, limit: int = 60) -> List[Dict[str, Any]]:
    out = []
    for p in _dir().glob("*.json"):
        try:
            run = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if paper_id and (run.get("meta") or {}).get("paper_id") != paper_id:
            continue
        out.append(summary_of(run))
    return sorted(out, key=lambda r: -(r.get("created_at") or 0))[:limit]


def load_history(run_id: str) -> Optional[Dict[str, Any]]:
    if not _ID.match(run_id):
        return None
    p = _dir() / f"{run_id}.json"
    try:
        run = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return {**run, "events": run.get("events", []), "total_events": len(run.get("events", []))}
