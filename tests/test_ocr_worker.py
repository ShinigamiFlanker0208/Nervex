"""OCR isolation: result passing, error propagation, timeout kill (child targets are fakes; fork ctx for speed)."""
import time
import pytest
from ingestion import ocr_worker as w


def ok_target(conn, path, cuda, dev, frac):
    conn.send(("ok", {"raw_markdown": f"md:{path}:{frac}", "metadata": {}, "tables": [], "images": {}})); conn.close()

def err_target(conn, *a):
    conn.send(("err", "ValueError: boom", "tb")); conn.close()

def die_target(conn, *a):
    import os; os._exit(7)

def hang_target(conn, *a):
    time.sleep(60)

def big_target(conn, *a):
    conn.send(("ok", {"raw_markdown": "x" * 5_000_000, "metadata": {}, "tables": [], "images": {}})); conn.close()


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    monkeypatch.setattr(w, "remove_stale_ocr_servers", lambda: 0)


def run(target, **kw):
    return w.run_ocr_isolated("a.pdf", target=target, mp_context="fork", timeout=kw.pop("timeout", 20), **kw)


def test_result_roundtrip_and_gpu_fraction_forwarded():
    assert run(ok_target, gpu_fraction=0.4)["raw_markdown"] == "md:a.pdf:0.4"

def test_child_exception_becomes_runtime_error_with_message():
    with pytest.raises(RuntimeError, match="boom"):
        run(err_target)

def test_child_hard_crash_is_reported_not_hung():
    with pytest.raises(RuntimeError, match="exit code 7"):
        run(die_target)

def test_hang_is_killed_at_timeout():
    t = time.time()
    with pytest.raises(TimeoutError):
        run(hang_target, timeout=1)
    assert time.time() - t < 20

def test_large_payload_does_not_deadlock():
    assert len(run(big_target)["raw_markdown"]) == 5_000_000


def test_ocr_cache_hit_skips_ocr_and_miss_populates(tmp_path, monkeypatch):
    monkeypatch.setattr(w.settings, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-same")
    calls = []
    monkeypatch.setattr(w, "run_ocr_isolated", lambda *a, **k: calls.append(1) or {"raw_markdown": "hello", "metadata": {}})
    p = w.IsolatedPDFParser()
    assert p.parse_pdf(pdf)["raw_markdown"] == "hello" and p.parse_pdf(pdf)["raw_markdown"] == "hello"
    assert len(calls) == 1                                  # second call served from cache
    pdf.write_bytes(b"%PDF-different"); p.parse_pdf(pdf)
    assert len(calls) == 2                                  # new content -> new key


def test_empty_ocr_result_is_not_cached_and_corrupt_cache_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(w.settings, "OCR_CACHE_DIR", tmp_path / "cache")
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-x")
    calls = []
    monkeypatch.setattr(w, "run_ocr_isolated", lambda *a, **k: calls.append(1) or {"raw_markdown": "  "})
    p = w.IsolatedPDFParser(); p.parse_pdf(pdf); p.parse_pdf(pdf)
    assert len(calls) == 2
    from ingestion.pipeline import compute_paper_id
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "cache" / f"{compute_paper_id(pdf)}.json").write_text("{not json")
    p.parse_pdf(pdf)                                         # must not raise
