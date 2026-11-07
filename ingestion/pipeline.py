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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ProgressFn = Callable[[str], None]


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

    say(f"extracting entities from {len(chunks)} chunks")
    schemas, failed = [], 0
    for i, chunk in enumerate(chunks):
        try:
            schemas.append(extractor.extract_from_chunk(chunk["content"], _vault_for_chunk(chunk, vault)))
        except Exception as exc:  # LLM/validation failures are per-chunk, not fatal
            failed += 1
            say(f"chunk {i} extraction failed: {type(exc).__name__}: {exc}")
    if not schemas:
        raise RuntimeError(f"Entity extraction failed for all {len(chunks)} chunks")

    merged = extractor.merge_subgraph_schemas(schemas)

    say("persisting")
    graph_counts = graph_db.insert_paper_schema(merged)
    vector_db.add_chunks(chunks)

    page_count = meta.get("page_stats") or meta.get("pages") or []
    num_pages = len(page_count) if hasattr(page_count, "__len__") else int(page_count or 0)

    return {
        "paper_id": paper_id,
        "paper_title": merged.paper_title or title_guess,
        "num_pages": num_pages,
        "num_entities_extracted": graph_counts.get("nodes", 0),
        "num_relationships_merged": graph_counts.get("edges", 0),
        "status_message": f"ok: {len(schemas)}/{len(chunks)} chunks extracted"
        + (f", {failed} failed" if failed else ""),
    }
