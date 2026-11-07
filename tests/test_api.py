"""API contract tests with deps monkeypatched (no Neo4j / Docker / LLM)."""
import asyncio
import pytest
from fastapi.testclient import TestClient

from api import deps, routes
from api.jobs import jobs
import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(config.settings, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main, "_probe_subsystems", lambda: _async({"neo4j": True, "docker": True, "sandbox_image": True, "cuda": False}))
    return TestClient(main.app)


async def _async(v): return v


PDF = b"%PDF-1.4\n" + b"x" * 100


def test_rejects_non_pdf_extension(client):
    r = client.post("/api/v1/ingest/pdf", files={"file": ("a.txt", PDF)})
    assert r.status_code == 400


def test_rejects_fake_pdf_by_magic_bytes(client, tmp_path):
    r = client.post("/api/v1/ingest/pdf", files={"file": ("a.pdf", b"not a pdf at all")})
    assert r.status_code == 400
    assert list(tmp_path.glob("*.pdf")) == []          # rejected upload is not left on disk


def test_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr(config.settings, "MAX_UPLOAD_MB", 0)
    assert client.post("/api/v1/ingest/pdf", files={"file": ("a.pdf", PDF)}).status_code == 413


def test_upload_traversal_filename_is_not_used_as_path(client, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "_run_ingestion", lambda *a: None)
    r = client.post("/api/v1/ingest/pdf", files={"file": ("../../evil.pdf", PDF)})
    assert r.status_code == 202
    saved = list(tmp_path.glob("*.pdf"))
    assert len(saved) == 1 and "evil" not in saved[0].name


def test_ingest_job_lifecycle_success(client, monkeypatch):
    import ingestion.pipeline as pl
    monkeypatch.setattr(pl, "ingest_pdf", lambda *a, **k: {
        "paper_id": "p", "paper_title": "T", "num_pages": 1, "num_entities_extracted": 2,
        "num_relationships_merged": 1, "status_message": "ok"})
    for n in ("get_pdf_parser", "get_masker", "get_chunker", "get_extractor", "get_graph_db", "get_vector_db"):
        monkeypatch.setattr(deps, n, lambda: object())
    jid = client.post("/api/v1/ingest/pdf", files={"file": ("a.pdf", PDF)}).json()["job_id"]
    body = client.get(f"/api/v1/ingest/status/{jid}").json()   # TestClient runs background tasks before returning
    assert body["state"] == "done" and body["result"]["paper_id"] == "p"


def test_ingest_job_failure_is_recorded(client, monkeypatch):
    import ingestion.pipeline as pl
    def boom(*a, **k): raise RuntimeError("ocr died")
    monkeypatch.setattr(pl, "ingest_pdf", boom)
    for n in ("get_pdf_parser", "get_masker", "get_chunker", "get_extractor", "get_graph_db", "get_vector_db"):
        monkeypatch.setattr(deps, n, lambda: object())
    jid = client.post("/api/v1/ingest/pdf", files={"file": ("a.pdf", PDF)}).json()["job_id"]
    body = client.get(f"/api/v1/ingest/status/{jid}").json()
    assert body["state"] == "failed" and "ocr died" in body["error"]


def test_unknown_job_404(client):
    assert client.get("/api/v1/ingest/status/nope").status_code == 404


def test_graph_query_requires_node_and_bounds_depth(client):
    assert client.post("/api/v1/graph/query", json={}).status_code == 422
    assert client.post("/api/v1/graph/query", json={"node_id": "a", "depth": 99}).status_code == 422


def test_graph_query_ok_and_db_error_maps_to_503(client, monkeypatch):
    class G:
        def query_graph_neighborhood(self, n, d, pid=None): return {"nodes": [{"id": "1"}], "edges": []}
    monkeypatch.setattr(deps, "get_graph_db", lambda: G())
    r = client.post("/api/v1/graph/query", json={"node_id": "a"})
    assert r.status_code == 200 and r.json()["nodes"] == [{"id": "1"}]

    class Bad:
        def query_graph_neighborhood(self, n, d, pid=None): raise ConnectionError("down")
    monkeypatch.setattr(deps, "get_graph_db", lambda: Bad())
    assert client.post("/api/v1/graph/query", json={"node_id": "a"}).status_code == 503


def test_generate_code_maps_terminal_state(client, monkeypatch):
    class O:
        async def run_translation_loop(self, st):
            return {**st, "current_code": "print(1)", "attempt_count": 2, "execution_status": "SUCCESS",
                    "tensor_shape_traces": {"fc": {"input": "[1]", "output": "[1]"}}}
    monkeypatch.setattr(deps, "get_orchestrator", lambda: O())
    r = client.post("/api/v1/generate/code", json={"paper_id": "p", "target_component": "Attn"})
    b = r.json()
    assert r.status_code == 200 and b["execution_status"] == "SUCCESS" and b["tensor_shape_traces"]["fc"]


def test_transpile_unsupported_framework_is_422(client, monkeypatch):
    class T:
        async def transpile_model(self, *a): raise ValueError("Unsupported framework")
    monkeypatch.setattr(deps, "get_transpiler", lambda: T())
    r = client.post("/api/v1/features/transpile", json={
        "source_code": "x", "source_framework": "tf", "target_framework": "JAX", "architecture_skeleton": ""})
    assert r.status_code == 422


def test_ablation_rejects_baseline_that_fails_sandbox(client, monkeypatch):
    import sandbox.docker_controller as dc
    async def fake(state): return {"execution_status": "RUNTIME_ERROR", "execution_logs": [{"traceback": "boom"}]}
    monkeypatch.setattr(dc, "execute_script_in_sandbox", fake)
    r = client.post("/api/v1/features/ablation", json={
        "paper_id": "p", "target_component": "A", "baseline_code": "x", "expected_inputs": {"x": "(1,1)"},
        "mutation_prompt": "remove residual"})
    assert r.status_code == 422 and "boom" in r.json()["detail"]


def test_health_reports_degraded_when_a_required_subsystem_is_down(client, monkeypatch):
    import main
    monkeypatch.setattr(main, "_probe_subsystems", lambda: _async({"neo4j": False, "docker": True, "sandbox_image": True, "cuda": False}))
    b = client.get("/").json()
    assert b["status"] == "degraded" and "neo4j" in b["message"]


def test_cors_is_not_wildcard(client):
    r = client.options("/", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "evil.example" not in (r.headers.get("access-control-allow-origin") or "")


def test_graph_query_dedupes_edges_from_overlapping_paths(client, monkeypatch):
    class G:
        def query_graph_neighborhood(self, n, d, pid=None):
            e = {"id": "e1", "type": "X", "start": "a", "end": "b", "properties": {}}
            return {"nodes": [], "edges": [e, dict(e), {**e, "id": "e2"}]}
    monkeypatch.setattr(deps, "get_graph_db", lambda: G())
    assert len(client.post("/api/v1/graph/query", json={"node_id": "a", "depth": 2}).json()["edges"]) == 2


def test_generate_code_surfaces_orchestrator_failure_reason(client, monkeypatch):
    """Regression: a 401 inside the graph came back as a bare CRITICAL_FAILURE with no reason."""
    class O:
        async def run_translation_loop(self, st):
            return {**st, "execution_status": "CRITICAL_FAILURE", "error_type": "UNKNOWN_EXCEPTION",
                    "error_details": "[401] Unauthorized"}
    monkeypatch.setattr(deps, "get_orchestrator", lambda: O())
    b = client.post("/api/v1/generate/code", json={"paper_id": "p", "target_component": "A"}).json()
    assert b["execution_status"] == "CRITICAL_FAILURE" and "401" in b["error_details"]


def test_uploaded_pdf_is_deleted_after_ingestion_success_and_failure(client, tmp_path, monkeypatch):
    import ingestion.pipeline as pl
    for n in ("get_pdf_parser", "get_masker", "get_chunker", "get_extractor", "get_graph_db", "get_vector_db"):
        monkeypatch.setattr(deps, n, lambda: object())
    monkeypatch.setattr(pl, "ingest_pdf", lambda *a, **k: {"paper_id": "p", "paper_title": "T", "num_pages": 1,
                        "num_entities_extracted": 1, "num_relationships_merged": 0, "status_message": "ok"})
    client.post("/api/v1/ingest/pdf", files={"file": ("a.pdf", PDF)})
    assert list(tmp_path.glob("*.pdf")) == []
    def boom(*a, **k): raise RuntimeError("x")
    monkeypatch.setattr(pl, "ingest_pdf", boom)
    client.post("/api/v1/ingest/pdf", files={"file": ("a.pdf", PDF)})
    assert list(tmp_path.glob("*.pdf")) == []
