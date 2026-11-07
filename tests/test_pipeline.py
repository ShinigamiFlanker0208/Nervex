"""Ingestion pipeline with fakes: no GPU, no LLM, no DB."""
from types import SimpleNamespace
import pytest
from ingestion.pipeline import ingest_pdf, compute_paper_id


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    import ingestion.pipeline as pl
    monkeypatch.setattr(pl, "RETRY_BACKOFF_S", 0)


class FakeParser:
    def parse_pdf(self, p): return {"raw_markdown": "# T\nbody $x$", "metadata": {"page_stats": [1, 2, 3]}}

class FakeMasker:
    def mask_document(self, md): return "# T\nbody <MATH_BLOCK_0000>", {"<MATH_BLOCK_0000>": SimpleNamespace(original_latex="x")}

class FakeChunker:
    def __init__(self, n=2): self.n = n
    def chunk_markdown(self, m, meta):
        return [{"chunk_id": f"chunk_{i:04d}", "content": f"c{i}", "math_tokens_present": ["<MATH_BLOCK_0000>"], "metadata": meta}
                for i in range(self.n)]

class FakeExtractor:
    def __init__(self, fail_on=()): self.fail_on, self.seen = set(fail_on), []
    def extract_from_chunk(self, text, vault):
        self.seen.append(vault)
        if text in self.fail_on: raise RuntimeError("llm blew up")
        return SimpleNamespace(paper_title="P", nodes=[1], edges=[])
    def merge_subgraph_schemas(self, s): return SimpleNamespace(paper_title="Merged", nodes=s, edges=[])

class FakeGraph:
    def insert_paper_schema(self, s): return {"nodes": len(s.nodes), "edges": 0}

class FakeVec:
    def __init__(self): self.chunks = None
    def add_chunks(self, c): self.chunks = c


def run(tmp_path, extractor=None, chunker=None):
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-1.4 hello")
    vec = FakeVec()
    out = ingest_pdf(pdf, FakeParser(), FakeMasker(), chunker or FakeChunker(), extractor or FakeExtractor(),
                     FakeGraph(), vec)
    return out, vec, pdf


def test_happy_path_and_response_shape(tmp_path):
    out, vec, _ = run(tmp_path)
    assert out["paper_title"] == "P" and out["num_pages"] == 3 and out["num_entities_extracted"] == 2


def test_chunk_ids_are_paper_scoped(tmp_path):
    out, vec, _ = run(tmp_path)
    assert all(c["chunk_id"].startswith(out["paper_id"] + "_chunk_") for c in vec.chunks)


def test_vault_filtered_to_chunk_tokens(tmp_path):
    ex = FakeExtractor(); run(tmp_path, ex)
    assert ex.seen[0] == {"<MATH_BLOCK_0000>": "x"}


def test_partial_failure_is_reported_not_fatal(tmp_path):
    out, _, _ = run(tmp_path, FakeExtractor(fail_on={"c1"}))
    assert "1 failed" in out["status_message"]


def test_total_failure_fails_loud(tmp_path):
    with pytest.raises(RuntimeError):
        run(tmp_path, FakeExtractor(fail_on={"c0", "c1"}))


def test_zero_chunks_fails_loud(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, chunker=FakeChunker(0))


def test_paper_id_is_content_hash(tmp_path):
    a = tmp_path / "a.pdf"; b = tmp_path / "b.pdf"
    a.write_bytes(b"same"); b.write_bytes(b"same")
    assert compute_paper_id(a) == compute_paper_id(b)
    b.write_bytes(b"diff"); assert compute_paper_id(a) != compute_paper_id(b)


def test_parallel_extraction_preserves_order_and_reports_progress(tmp_path):
    import time, random
    class Slow(FakeExtractor):
        def extract_from_chunk(self, text, vault):
            time.sleep(random.random() * 0.05)           # finish out of order
            return SimpleNamespace(paper_title="P", nodes=[text], edges=[])
    msgs = []
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-1.4 hello")
    ingest_pdf(pdf, FakeParser(), FakeMasker(), FakeChunker(8), Slow(), FakeGraph(), FakeVec(),
               on_progress=msgs.append, max_workers=4)
    assert msgs[-2] == "extracted 8/8" or "extracted 8/8" in msgs
    ex = Slow(); ex.merge_subgraph_schemas = lambda s: SimpleNamespace(paper_title="M", nodes=[n for x in s for n in x.nodes], edges=[])
    out = ingest_pdf(pdf, FakeParser(), FakeMasker(), FakeChunker(8), ex, type("G", (), {"insert_paper_schema": lambda s, m: captured.append(m.nodes) or {"nodes": 8, "edges": 0}})(), FakeVec(), max_workers=4)
    assert captured[0] == [f"c{i}" for i in range(8)]     # input order, not completion order

captured = []


def test_transient_failure_is_retried_then_succeeds(tmp_path, monkeypatch):
    import ingestion.pipeline as pl
    monkeypatch.setattr(pl, "RETRY_BACKOFF_S", 0)
    class Flaky(FakeExtractor):
        def __init__(self): super().__init__(); self.calls = 0
        def extract_from_chunk(self, text, vault):
            self.calls += 1
            if self.calls == 1: raise RuntimeError("[400] unknown field guided_json")
            return super().extract_from_chunk(text, vault)
    ex = Flaky()
    out, _, _ = run(tmp_path, ex, chunker=FakeChunker(1))
    assert ex.calls == 2 and "failed" not in out["status_message"]


def test_persistent_failure_exhausts_retries(tmp_path, monkeypatch):
    import ingestion.pipeline as pl
    monkeypatch.setattr(pl, "RETRY_BACKOFF_S", 0)
    ex = FakeExtractor(fail_on={"c1"})
    run(tmp_path, ex)
    assert sum(1 for _ in ex.seen) == 1 + pl.RETRIES        # c0 once, c1 RETRIES times


def test_title_majority_vote_beats_first_chunk():
    from ingestion.pipeline import pick_title
    mk = lambda t: SimpleNamespace(paper_title=t)
    assert pick_title([mk("Google Permission Notice"), mk("Attention Is All You Need"), mk("Attention Is All You Need")], "x") == "Attention Is All You Need"
    assert pick_title([mk(""), mk("  ")], "fallback") == "fallback"
