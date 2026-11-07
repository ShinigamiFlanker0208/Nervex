"""Upstream LLM failures: classification, retry, and clean API error mapping (live E2E: bare 500 on a socket timeout)."""
import asyncio
import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from agents import llm_retry as r
from agents.llm_retry import classify, ainvoke_with_retry


class SocketTimeoutError(Exception): pass

_real_sleep = asyncio.sleep


def _fast(monkeypatch):
    monkeypatch.setattr(r.asyncio, "sleep", lambda s: _real_sleep(0))


@pytest.mark.parametrize("exc,expect", [
    (SocketTimeoutError("Timeout on reading data from socket"), (504, "timeout")),
    (asyncio.TimeoutError(), (504, "timeout")),
    (Exception("[429] Too Many Requests"), (429, "rate_limited")),
    (Exception("[503] Service Unavailable"), (502, "upstream_error")),
    (Exception("[401] Unauthorized Invalid JWT"), (502, "upstream_auth")),
    (ConnectionError("refused"), (502, "connection")),
    (Exception("[400] unknown field guided_json"), (None, "")),       # our request is wrong: do not mask as upstream
    (ValueError("nope"), (None, "")),
])
def test_classify(exc, expect):
    assert classify(exc) == expect


class Flaky:
    def __init__(self, fails, exc): self.fails, self.exc, self.calls = fails, exc, 0
    async def ainvoke(self, m):
        self.calls += 1
        if self.calls <= self.fails: raise self.exc
        return SimpleNamespace(content="ok")


def run(c): return asyncio.run(c)


def test_retry_recovers_from_transient(monkeypatch):
    _fast(monkeypatch)
    llm = Flaky(2, SocketTimeoutError("Timeout on reading data from socket"))
    assert run(ainvoke_with_retry(llm, "x")).content == "ok" and llm.calls == 3


def test_non_transient_is_not_retried(monkeypatch):
    llm = Flaky(5, ValueError("bad prompt"))
    with pytest.raises(ValueError): run(ainvoke_with_retry(llm, "x"))
    assert llm.calls == 1


def test_gives_up_after_attempts(monkeypatch):
    _fast(monkeypatch)
    llm = Flaky(99, Exception("[429] Too Many Requests"))
    with pytest.raises(Exception, match="429"): run(ainvoke_with_retry(llm, "x"))
    assert llm.calls == 3


@pytest.fixture
def client(monkeypatch):
    import main
    from api import deps
    async def probe(): return {"neo4j": True, "docker": True, "sandbox_image": True, "cuda": True}
    monkeypatch.setattr(main, "_probe_subsystems", probe)
    return TestClient(main.app, raise_server_exceptions=False), deps


BODY = {"source_code": "x", "source_framework": "PyTorch", "target_framework": "JAX", "architecture_skeleton": ""}


@pytest.mark.parametrize("exc,status,kind", [
    (SocketTimeoutError("Timeout on reading data from socket"), 504, "timeout"),
    (Exception("[429] Too Many Requests"), 429, "rate_limited"),
    (Exception("[503] boom"), 502, "upstream_error"),
])
def test_api_maps_upstream_failures_to_typed_json(client, monkeypatch, exc, status, kind):
    c, deps = client
    class T:
        async def transpile_model(self, *a): raise exc
    monkeypatch.setattr(deps, "get_transpiler", lambda: T())
    res = c.post("/api/v1/features/transpile", json=BODY)
    assert res.status_code == status and res.json()["error"] == kind
    if status in (429, 504): assert res.headers["retry-after"]


def test_unknown_exception_is_json_500_without_leaking_message(client, monkeypatch):
    c, deps = client
    class T:
        async def transpile_model(self, *a): raise RuntimeError("secret internal path /home/x")
    monkeypatch.setattr(deps, "get_transpiler", lambda: T())
    res = c.post("/api/v1/features/transpile", json=BODY)
    assert res.status_code == 500 and res.json()["error"] == "internal_error" and "secret" not in res.text
