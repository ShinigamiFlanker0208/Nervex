"""
Ingestion Pipeline Orchestrator (PDF -> Neo4j + ChromaDB)

Why this exists: the four ingestion stages (parse, mask, chunk, extract) were only wired together in
a throwaway script. The API needs ONE callable that does the whole thing, reports progress, and can be
tested without GPUs or an LLM, so every collaborator is injected.

Data flow:
    PDF -> PDFIngestionEngine.parse_pdf -> MathMasker.mask_document -> SemanticChunker.chunk_markdown
        -> OntologyExtractor.extract_from_chunk (per chunk) -> merge_subgraph_schemas
        -> Neo4jGraphManager.insert_paper_schema   +   VectorStoreManager.add_chunks

Catastrophes this module prevents:
  * Second paper silently overwrites the first one's vectors (SemanticChunker ids are `chunk_0000`...).
    -> every chunk id is prefixed with the deterministic paper_id.
  * One flaky LLM call kills a 40-chunk ingestion.  -> per-chunk failures are counted, not fatal,
    but if EVERY chunk fails we raise (fail loud) instead of reporting an empty "success".
  * Re-uploading the same PDF duplicates data.  -> paper_id is a content hash; graph uses MERGE and
    Chroma uses upsert, so re-ingestion is idempotent.
"""
import hashlib
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ProgressFn = Callable[[str], None]
RETRIES = 6
RETRY_BACKOFF_S = 2.0           # transient (non-rate-limit) errors: 2s, 4s, ...
RATE_LIMIT_BASE_S = 8.0         # 429: 8s, 16s, 32s, 60s (capped), shared by ALL workers
RATE_LIMIT_MAX_S = 60.0


def is_rate_limited(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def pick_title(schemas: List[Any], fallback: str) -> str:
    """Majority vote over per-chunk titles. The first chunk is often a copyright/permission notice,
    so trusting schemas[0] (as the merge does) can label a paper 'Google Permission Notice...'."""
    votes = Counter(t.strip() for t in (getattr(x, "paper_title", "") for x in schemas) if t and t.strip())
    return votes.most_common(1)[0][0] if votes else fallback


def compute_paper_id(pdf_path: Path) -> str:
    """Deterministic id = first 16 hex chars of the SHA-256 of the file bytes."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _vault_for_chunk(chunk: Dict[str, Any], vault: Dict[str, Any]) -> Dict[str, str]:
    """Only hand the LLM the formulas this chunk actually references (keeps the prompt small)."""
    return {
        tok: vault[tok].original_latex
        for tok in chunk.get("math_tokens_present", [])
        if tok in vault
    }


def ingest_pdf(
    pdf_path: Path,
    parser: Any,
    masker: Any,
    chunker: Any,
    extractor: Any,
    graph_db: Any,
    vector_db: Any,
    on_progress: Optional[ProgressFn] = None,
    max_workers: int = 1,
    max_failed_fraction: float = 0.25,
) -> Dict[str, Any]:
    """Run the full ingestion. Returns a dict matching api.schemas.IngestionResponse."""
    say = on_progress or (lambda _msg: None)

    paper_id = compute_paper_id(pdf_path)

    say("parsing")
    parsed = parser.parse_pdf(pdf_path)
    markdown: str = parsed["raw_markdown"]
    meta = parsed.get("metadata") or {}
    if not markdown.strip():
        raise ValueError("OCR produced empty markdown; is this a scanned/corrupt PDF?")

    say("masking")
    masked, vault = masker.mask_document(markdown)

    say("chunking")
    title_guess = pdf_path.stem
    chunks: List[Dict[str, Any]] = chunker.chunk_markdown(
        masked, {"source": title_guess, "paper_id": paper_id}
    )
    if not chunks:
        raise ValueError("Chunker returned 0 chunks")
    for c in chunks:
        c["chunk_id"] = f"{paper_id}_{c['chunk_id']}"  # paper-scoped, see module docstring

    say(f"extracting entities from {len(chunks)} chunks ({max_workers} parallel)")
    done, lock = [0], threading.Lock()
    cooldown_until = [0.0]          # monotonic deadline; one 429 pauses every worker, not just the unlucky one

    def work(i_chunk):
        i, chunk = i_chunk
        out, err = None, None
        for attempt in range(1, RETRIES + 1):
            wait = cooldown_until[0] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                out = extractor.extract_from_chunk(chunk["content"], _vault_for_chunk(chunk, vault))
                err = None
                break
            except Exception as exc:  # LLM/validation failures are per-chunk, not fatal
                err = f"chunk {i} extraction failed (attempt {attempt}/{RETRIES}): {type(exc).__name__}: {exc}"
                if attempt < RETRIES:
                    if is_rate_limited(exc):
                        delay = min(RATE_LIMIT_MAX_S, RATE_LIMIT_BASE_S * 2 ** (attempt - 1)) * (0.75 + random.random() / 2)
                        with lock:
                            cooldown_until[0] = max(cooldown_until[0], time.monotonic() + delay)
                    else:
                        time.sleep(RETRY_BACKOFF_S * attempt)   # NIM backends are occasionally flaky (seen: 400 on 1/68)
        with lock:
            done[0] += 1
            say(err or f"extracted {done[0]}/{len(chunks)}")
        return out

    # pool.map yields results in INPUT order, so the merge (first-seen node wins) is deterministic
    # regardless of which LLM call finishes first.
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = list(pool.map(work, enumerate(chunks)))
    schemas = [r for r in results if r is not None]
    failed = len(results) - len(schemas)
    if not schemas:
        raise RuntimeError(f"Entity extraction failed for all {len(chunks)} chunks")
    # Fail loud BEFORE touching the stores: a paper missing most of its content must not replace a good graph
    # (measured: a 429 storm once left 19/68 chunks and the job still said "done").
    if failed / len(chunks) > max_failed_fraction:
        raise RuntimeError(
            f"Extraction too lossy: {failed}/{len(chunks)} chunks failed (limit {max_failed_fraction:.0%}). "
            "Nothing was written. Likely rate-limited: lower EXTRACTION_CONCURRENCY or retry later.")

    merged = extractor.merge_subgraph_schemas(schemas)

    say("persisting")
    graph_counts = graph_db.replace_paper_schema(merged, paper_id)   # atomic, idempotent per paper
    vector_db.add_chunks(chunks)

    page_count = meta.get("page_stats") or meta.get("pages") or []
    num_pages = len(page_count) if hasattr(page_count, "__len__") else int(page_count or 0)

    return {
        "paper_id": paper_id,
        "paper_title": pick_title(schemas, merged.paper_title or title_guess),
        "num_pages": num_pages,
        "num_entities_extracted": graph_counts.get("nodes", 0),
        "num_relationships_merged": graph_counts.get("edges", 0),
        "status_message": f"ok: {len(schemas)}/{len(chunks)} chunks extracted"
        + (f", {failed} failed" if failed else ""),
    }
