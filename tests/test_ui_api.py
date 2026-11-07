"""UI read models + observable runs (fakes only)."""
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import deps
from api.runs import RunStore, summarize


@pytest.fixture(autouse=True)
def _runs_dir(tmp_path, monkeypatch):
    # finished runs are persisted; never let a test write into the real data/runs
    from config import settings
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path / "runs")


@pytest.fixture
def client(monkeypatch):
    import main
    async def probes(): return {"neo4j": True, "docker": True, "sandbox_image": True, "cuda": False}
    monkeypatch.setattr(main, "_probe_subsystems", probes)
    return TestClient(main.app)


class FakeDB:
    def list_papers(self): return [{"paper_id": "p1", "title": "T", "nodes": 3, "edges": 2}]
    def paper_overview(self, pid, limit, include=None):
        return {"nodes": [{"id": "a", "name": "A", "type": "Concept", "description": None, "degree": 2}] if pid == "p1" else [],
                "edges": []}
    def top_components(self, pid, limit): return [{"id": "mha", "name": "MHA", "description": "x", "degree": 9, "formulas": 3}]


def test_papers_graph_components(client, monkeypatch):
    monkeypatch.setattr(deps, "get_graph_db", lambda: FakeDB())
    assert client.get("/api/v1/papers").json()[0]["title"] == "T"
    assert client.get("/api/v1/papers/p1/graph").json()["nodes"][0]["id"] == "a"
    assert client.get("/api/v1/papers/nope/graph").status_code == 404
    assert client.get("/api/v1/papers/p1/graph?include=x,y").status_code == 200
    assert client.get("/api/v1/papers/p1/components").json()[0]["id"] == "mha"


def test_summarize_pulls_node_ids_out_of_formulas():
    ev = summarize("researcher", {"relevant_formulas": ["[sdpa] \\softmax(QK^T)", "[mha] concat"], "hyperparameters": {"h": 8}})
    assert ev["node_ids"] == ["sdpa", "mha"] and ev["hyperparameters"] == ["h"]


def test_summarize_sandbox_hides_model_boundary_layer():
    ev = summarize("sandbox_executor", {"execution_status": "SUCCESS", "attempt_count": 2,
                                        "tensor_shape_traces": {"w_q": {}, "__model__": {}}, "execution_logs": []})
    assert ev["layers"] == ["w_q"] and ev["status"] == "SUCCESS"


def test_store_returns_only_new_events():
    s = RunStore(); rid = s.create({})
    s.event(rid, {"node": "a"}); s.event(rid, {"node": "b"})
    assert [e["node"] for e in s.get(rid, 1)["events"]] == ["b"] and s.get(rid, 1)["total_events"] == 2


class Workflow:
    async def astream(self, state, config=None, stream_mode=None):
        yield {"researcher": {"relevant_formulas": ["[x] y"], "hyperparameters": {}}}
        yield {"coder": {"current_code": "a = 1", "attempt_count": 1}}
        yield {"sandbox_executor": {"execution_status": "SUCCESS", "attempt_count": 1, "tensor_shape_traces": {"fc": {}}}}


def test_run_streams_events_and_finishes(client, monkeypatch):
    monkeypatch.setattr(deps, "get_orchestrator", lambda: SimpleNamespace(workflow=Workflow()))
    rid = client.post("/api/v1/generate/runs", json={"paper_id": "p", "target_component": "c"}).json()["run_id"]
    for _ in range(50):
        r = client.get(f"/api/v1/generate/runs/{rid}").json()
        if r["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert r["state"] == "done" and r["result"]["generated_code"] == "a = 1"
    assert [e["node"] for e in r["events"]] == ["start", "researcher", "coder", "sandbox_executor", "end"]
    assert client.get(f"/api/v1/generate/runs/{rid}?since=4").json()["events"][0]["node"] == "end"


def test_run_crash_is_reported_not_lost(client, monkeypatch):
    class Boom:
        async def astream(self, *a, **k):
            raise RuntimeError("kaboom")
            yield
    monkeypatch.setattr(deps, "get_orchestrator", lambda: SimpleNamespace(workflow=Boom()))
    rid = client.post("/api/v1/generate/runs", json={"paper_id": "p", "target_component": "c"}).json()["run_id"]
    for _ in range(50):
        r = client.get(f"/api/v1/generate/runs/{rid}").json()
        if r["state"] != "queued" and r["state"] != "running":
            break
        time.sleep(0.05)
    assert r["state"] == "failed" and "kaboom" in r["result"]["error_details"]


def test_unknown_run_404(client):
    assert client.get("/api/v1/generate/runs/zzz").status_code == 404


# ---------------- persistence + system config ----------------
def test_finished_run_is_saved_listed_and_reloadable(client, monkeypatch, tmp_path):
    from config import settings
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(deps, "get_orchestrator", lambda: SimpleNamespace(workflow=Workflow()))
    rid = client.post("/api/v1/generate/runs", json={"paper_id": "p9", "target_component": "mha"}).json()["run_id"]
    for _ in range(60):
        if client.get(f"/api/v1/generate/runs/{rid}").json()["state"] == "done":
            break
        time.sleep(0.05)
    time.sleep(0.1)
    hist = client.get("/api/v1/generate/history?paper_id=p9").json()
    assert hist[0]["run_id"] == rid and hist[0]["status"] == "SUCCESS" and hist[0]["layers"] == 1
    assert client.get("/api/v1/generate/history?paper_id=other").json() == []
    full = client.get(f"/api/v1/generate/history/{rid}").json()
    assert full["result"]["generated_code"] == "a = 1" and full["events"][-1]["node"] == "end"


def test_history_id_cannot_escape_the_runs_dir(client):
    assert client.get("/api/v1/generate/history/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/v1/generate/history/not-hex").status_code == 404


def test_system_config_has_real_limits_and_no_secret(client):
    r = client.get("/api/v1/system/config").json()
    assert r["sandbox"]["network"] == "none" and r["loop"]["max_attempts"] >= 1 and "api_key" not in str(r).lower()


def test_researcher_event_carries_latex():
    ev = summarize("researcher", {"relevant_formulas": ["[eq1] $a+b$"], "hyperparameters": {}})
    assert ev["formulas_tex"] == [{"id": "eq1", "latex": "$a+b$"}]


def test_tests_do_not_touch_real_runs_dir():
    from config import settings
    assert "pytest" in str(settings.RUNS_DIR) or "tmp" in str(settings.RUNS_DIR)
