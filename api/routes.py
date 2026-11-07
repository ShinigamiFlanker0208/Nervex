"""
FastAPI Router Endpoint Definitions

Exposes HTTP API endpoints for frontend interaction, wiring typed request schemas
to ingestion, the LangGraph orchestrator, the sandbox, and the explainer / feature engines.

Contracts at the boundary (senior principle): every body is a Pydantic model, every heavy
dependency comes from `api.deps` (lazy + monkeypatchable), every failure maps to a real HTTP code.
"""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from api import deps
from api.jobs import jobs
from api.schemas import (
    AblationStudyRequest, AblationStudyResponse,
    CodeGenerationRequest, CodeGenerationResponse,
    ComponentSummary, ExplainerChatRequest, ExplainerChatResponse, GraphOverviewResponse, PaperSummary, RunAccepted, RunStatus,
    GraphNeighborhoodResponse, IngestionJobAccepted, IngestionJobStatus,
    KnowledgeGraphQueryRequest,
    TranspilationRequest, TranspilationResponse,
)
from config import settings

log = logging.getLogger("nervex.api")
router = APIRouter(tags=["SoulGraph Endpoints"])


# =====================================================================
# Phase 1: Document Ingestion & Knowledge Graph
# =====================================================================

def _run_ingestion(job_id: str, pdf_path: Path) -> None:
    """Background worker (sync -> FastAPI runs it in a threadpool). Never raises: errors land in the job."""
    from ingestion.pipeline import ingest_pdf

    jobs.update(job_id, state="running")
    try:
        with deps.ingestion_lock:  # one OCR+LLM ingestion at a time (GPU/VRAM, API rate limits)
            result = ingest_pdf(
                pdf_path,
                parser=deps.get_pdf_parser(),
                masker=deps.get_masker(),
                chunker=deps.get_chunker(),
                extractor=deps.get_extractor(),
                graph_db=deps.get_graph_db(),
                vector_db=deps.get_vector_db(),
                on_progress=lambda m: jobs.log(job_id, m),
                max_workers=settings.EXTRACTION_CONCURRENCY,
                max_failed_fraction=settings.EXTRACTION_MAX_FAILED_FRACTION,
            )
        jobs.update(job_id, state="done", result=result)
    except Exception as exc:  # fail loud into the job record, with the type for debugging
        log.exception("ingestion failed")
        jobs.update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # The upload is only an input: results live in Neo4j/Chroma and the OCR cache is content-addressed,
        # so keeping the PDF just leaks disk (uploads/ grew by one file per request).
        pdf_path.unlink(missing_ok=True)


@router.post("/ingest/pdf", status_code=status.HTTP_202_ACCEPTED, response_model=IngestionJobAccepted,
             summary="Upload a research paper PDF; returns a job to poll")
async def ingest_academic_pdf(background: BackgroundTasks, file: UploadFile = File(...)) -> IngestionJobAccepted:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf files are supported.")

    # Never trust the client filename for the path (traversal): store under a random name.
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = settings.UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    size, first = 0, b""
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):
                if not first:
                    first = chunk[:5]
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                        f"File exceeds {settings.MAX_UPLOAD_MB} MB.")
                out.write(chunk)
        if not first.startswith(b"%PDF"):  # extension is not proof; check magic bytes
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a valid PDF.")
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    job_id = jobs.create(filename)
    background.add_task(_run_ingestion, job_id, dest)
    return IngestionJobAccepted(job_id=job_id, status_url=f"/api/v1/ingest/status/{job_id}")


@router.get("/ingest/status/{job_id}", response_model=IngestionJobStatus, summary="Poll an ingestion job")
async def ingestion_status(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job_id.")
    return job


@router.post("/graph/query", response_model=GraphNeighborhoodResponse,
             summary="Query the Neo4j knowledge graph around a node")
async def query_knowledge_graph(request: KnowledgeGraphQueryRequest) -> GraphNeighborhoodResponse:
    if not request.node_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "node_id is required.")
    import asyncio
    try:
        sub = await asyncio.to_thread(deps.get_graph_db().query_graph_neighborhood, request.node_id, request.depth, request.paper_id)
    except Exception as exc:
        log.exception("graph query failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Graph database error: {type(exc).__name__}")
    # The manager returns one row per PATH, so the same edge repeats at depth >= 2: dedupe by id.
    edges = list({e["id"]: e for e in sub["edges"]}.values())
    return GraphNeighborhoodResponse(center_node_id=request.node_id, nodes=sub["nodes"], edges=edges)


# =====================================================================
# Phase 2 & 3: Code Generation + Sandbox
# =====================================================================

def initial_state_for(req: CodeGenerationRequest) -> Dict[str, Any]:
    return {
        "paper_id": req.paper_id,
        "target_component": req.target_component,
        "target_framework": req.target_framework,
        "paper_context": {},
        "relevant_formulas": [],
        "hyperparameters": {},
        "structural_skeleton": "",
        "expected_inputs": {},
        "current_code": "",
        "attempt_count": 0,
        "execution_status": "PENDING",
        "critic_feedback": "",
        "execution_logs": [],
        "tensor_shape_traces": {},
    }


@router.post("/generate/code", response_model=CodeGenerationResponse,
             summary="Run the Researcher->Architect->Coder->Sandbox->Critic loop")
async def generate_and_execute_code(request: CodeGenerationRequest) -> CodeGenerationResponse:
    async with deps.generation_lock:  # one GPU sandbox pipeline at a time
        orchestrator = deps.get_orchestrator()
        final = await orchestrator.run_translation_loop(initial_state_for(request))
    return CodeGenerationResponse(
        paper_id=request.paper_id,
        target_component=request.target_component,
        target_framework=request.target_framework,
        generated_code=final.get("current_code", ""),
        attempt_count=final.get("attempt_count", 0),
        execution_status=final.get("execution_status", "UNKNOWN"),
        tensor_shape_traces=final.get("tensor_shape_traces", {}) or {},
        error_logs=final.get("execution_logs", []) or [],
        error_type=final.get("error_type"),
        error_details=final.get("error_details"),
    )


# =====================================================================
# Phase 4: Explainer Chat
# =====================================================================

@router.post("/explainer/chat", response_model=ExplainerChatResponse,
             summary="Ask about highlighted generated code, grounded in the paper")
async def chat_explain_code(request: ExplainerChatRequest) -> ExplainerChatResponse:
    if request.highlighted_line_end < request.highlighted_line_start:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "highlighted_line_end < highlighted_line_start.")
    result = await deps.get_explainer().explain(
        request.user_question,
        (request.highlighted_line_start, request.highlighted_line_end),
        request.current_code_script,
        request.chat_history,
        paper_id=request.paper_id,
    )
    return ExplainerChatResponse(**result)


# =====================================================================
# Advanced features
# =====================================================================

@router.post("/features/ablation", response_model=AblationStudyResponse,
             summary="Mutate a working architecture, re-run in the sandbox, compare")
async def run_ablation_experiment(request: AblationStudyRequest) -> AblationStudyResponse:
    from features.ablation import AblationStudyController
    from sandbox.docker_controller import execute_script_in_sandbox
    # A baseline is only trustworthy if it ACTUALLY passes the sandbox. Verify it for real
    # (this also produces the baseline tensor-shape telemetry we compare against).
    async with deps.generation_lock:
        check = await execute_script_in_sandbox(
            {"current_code": request.baseline_code, "expected_inputs": request.expected_inputs})
    if check.get("execution_status") != "SUCCESS":
        tb = (check.get("execution_logs") or [{}])[-1].get("traceback", "")
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Baseline failed the sandbox:\n{tb[-1500:]}")
    baseline_state = {
        **initial_state_for(CodeGenerationRequest(paper_id=request.paper_id, target_component=request.target_component)),
        "current_code": request.baseline_code,
        "expected_inputs": request.expected_inputs,
        "execution_status": "SUCCESS",
        "tensor_shape_traces": check.get("tensor_shape_traces", {}),
    }
    async with deps.generation_lock:
        controller = AblationStudyController(deps.get_orchestrator(), sandbox_controller=None)
        try:
            report = await controller.run_ablation_experiment(baseline_state, request.mutation_prompt)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    resp = AblationStudyResponse(
        mutation_prompt=request.mutation_prompt,
        baseline_code=request.baseline_code,
        ablated_code=report["mutated_code"],
        baseline_metrics=report["baseline_metrics"],
        ablated_metrics=report["ablated_metrics"],
        diff_explanation=report["diff_summary"],
    )
    try:
        from api.runs import save_ablation
        save_ablation(request.paper_id, request.target_component, resp.model_dump())
    except Exception:
        log.exception("could not save ablation")
    return resp


@router.get("/features/ablations", summary="Saved ablation experiments for a paper (optionally one component)")
async def ablation_history(paper_id: str, component: str = "") -> List[Dict[str, Any]]:
    import asyncio
    from api.runs import list_ablations
    return await asyncio.to_thread(list_ablations, paper_id, component or None)


@router.get("/papers/{paper_id}/passages", summary="The paper's stored text, in reading order")
async def paper_passages(paper_id: str) -> List[Dict[str, Any]]:
    import asyncio
    try:
        rows = await asyncio.to_thread(deps.get_vector_db().paper_passages, paper_id)
    except Exception as exc:
        log.exception("passages failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Vector store error: {type(exc).__name__}")
    vault = await asyncio.to_thread(_math_vault, paper_id)
    tok = __import__("re").compile(r"<MATH_BLOCK_\d+>")
    for r in rows:   # chunks were stored MASKED; put the paper's real LaTeX back
        r["text"] = tok.sub(lambda m: vault.get(m.group(0), m.group(0)), r["text"])
    return [{"id": r["id"], "text": r["text"], "header": (r["meta"] or {}).get("header") or (r["meta"] or {}).get("section") or ""} for r in rows]


@router.post("/features/transpile", response_model=TranspilationResponse,
             summary="PyTorch <-> JAX/Equinox transpilation")
async def transpile_model_framework(request: TranspilationRequest) -> TranspilationResponse:
    try:
        out = await deps.get_transpiler().transpile_model(
            request.source_code, request.source_framework, request.target_framework, request.architecture_skeleton,
            request.expected_inputs,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    return TranspilationResponse(
        source_framework=out["source_framework"], target_framework=out["target_framework"],
        transpiled_code=out["transpiled_code"], syntax_valid=out["syntax_valid"], notes=out["notes"],
        runtime_verified=out.get("runtime_verified"), runtime_logs=out.get("runtime_logs", ""),
        output_shape=out.get("output_shape"),
    )


# =====================================================================
# UI read models: what has been ingested, and its drawable skeleton
# =====================================================================

async def _graph_call(fn, *args):
    import asyncio
    try:
        return await asyncio.to_thread(fn, *args)
    except Exception as exc:
        log.exception("graph read failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Graph database error: {type(exc).__name__}")


@router.get("/papers", response_model=List[PaperSummary], summary="Ingested papers")
async def list_papers() -> List[PaperSummary]:
    return [PaperSummary(**p) for p in await _graph_call(deps.get_graph_db().list_papers)]


@router.get("/papers/{paper_id}/graph", response_model=GraphOverviewResponse,
            summary="Best-connected entities of a paper and the edges among them")
async def paper_graph(paper_id: str, limit: int = 120, include: str = "") -> GraphOverviewResponse:
    ids = [i.strip() for i in include.split(",") if i.strip()][:60]
    sub = await _graph_call(deps.get_graph_db().paper_overview, paper_id, limit, ids)
    if not sub["nodes"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown paper_id.")
    return GraphOverviewResponse(paper_id=paper_id, **sub)


@router.get("/papers/{paper_id}/components", response_model=List[ComponentSummary],
            summary="Architecture components you can generate code for")
async def paper_components(paper_id: str, limit: int = 24) -> List[ComponentSummary]:
    return [ComponentSummary(**c) for c in await _graph_call(deps.get_graph_db().top_components, paper_id, limit)]


# =====================================================================
# Live generation runs (same loop as /generate/code, but observable)
# =====================================================================

async def _drive_run(run_id: str, req: CodeGenerationRequest) -> None:
    import asyncio
    import time
    from langgraph.errors import GraphRecursionError
    from api.runs import runs, summarize

    state = initial_state_for(req)
    err_type = err_details = None
    try:
        async with deps.generation_lock:                      # queued until the GPU is free
            runs.update(run_id, state="running", started_at=time.time())
            runs.event(run_id, {"node": "start"})
            orch = deps.get_orchestrator()
            async with asyncio.timeout(settings.TRANSLATION_TIMEOUT_SECONDS):
                async for chunk in orch.workflow.astream(state, config={"recursion_limit": 50}, stream_mode="updates"):
                    for node, update in chunk.items():
                        update = update or {}
                        state.update(update)
                        runs.event(run_id, summarize(node, update))
    except TimeoutError:
        err_type = "GLOBAL_TIMEOUT"
    except GraphRecursionError:
        err_type = "INFINITE_LOOP"
    except Exception as exc:
        log.exception("run failed")
        err_type, err_details = "UNKNOWN_EXCEPTION", f"{type(exc).__name__}: {exc}"
        state["execution_status"] = "CRITICAL_FAILURE"
    if err_type and state.get("execution_status") in (None, "PENDING", "RUNTIME_ERROR", "STATIC_ANALYSIS_FAILED"):
        state["execution_status"] = "FAILED"
    result = CodeGenerationResponse(
        paper_id=req.paper_id, target_component=req.target_component, target_framework=req.target_framework,
        generated_code=state.get("current_code", ""), attempt_count=state.get("attempt_count", 0),
        execution_status=state.get("execution_status", "UNKNOWN"),
        tensor_shape_traces=state.get("tensor_shape_traces", {}) or {},
        error_logs=state.get("execution_logs", []) or [], error_type=err_type, error_details=err_details)
    runs.event(run_id, {"node": "end", "status": result.execution_status})
    runs.update(run_id, state="done" if result.execution_status == "SUCCESS" else "failed", result=result.model_dump())
    try:
        from api.runs import persist
        persist(run_id)
    except Exception:                                          # saving history must never break the run itself
        log.exception("could not persist run %s", run_id)


@router.post("/generate/runs", status_code=status.HTTP_202_ACCEPTED, response_model=RunAccepted,
             summary="Start an observable generation run (poll /generate/runs/{id})")
async def start_run(request: CodeGenerationRequest) -> RunAccepted:
    import asyncio
    from api.runs import runs
    rid = runs.create({"paper_id": request.paper_id, "target_component": request.target_component,
                       "target_framework": request.target_framework})
    task = asyncio.create_task(_drive_run(rid, request))
    _RUN_TASKS.add(task)                                       # keep a reference or the task can be GC'd
    task.add_done_callback(_RUN_TASKS.discard)
    return RunAccepted(run_id=rid)


_RUN_TASKS: set = set()


@router.get("/generate/history", summary="Finished runs saved on disk (newest first)")
async def run_history(paper_id: str = "", limit: int = 60) -> List[Dict[str, Any]]:
    import asyncio
    from api.runs import list_history
    return await asyncio.to_thread(list_history, paper_id or None, max(1, min(limit, 200)))


@router.get("/generate/history/{run_id}", response_model=RunStatus, summary="One saved run with all its events")
async def run_history_item(run_id: str) -> Dict[str, Any]:
    from api.runs import load_history
    run = load_history(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown run_id.")
    return run


@router.get("/system/config", summary="The limits and settings the pipeline really runs with (no secrets)")
async def system_config() -> Dict[str, Any]:
    from agents.llm_factory import describe
    try:
        llm = describe()
    except ValueError as exc:
        llm = {"error": str(exc)}
    return {
        "llm": llm,
        "sandbox": {"network": "none", "capabilities": "all dropped", "user": "1000:1000", "workspace": "read-only",
                    "tmpfs_mb": 64, "memory_mb": settings.CONTAINER_MEMORY_LIMIT_MB, "timeout_s": settings.CONTAINER_TIMEOUT_SECONDS,
                    "image": settings.DOCKER_SANDBOX_IMAGE},
        "loop": {"max_attempts": settings.MAX_EXECUTION_ATTEMPTS, "loop_timeout_s": settings.TRANSLATION_TIMEOUT_SECONDS},
        "ingest": {"extraction_concurrency": settings.EXTRACTION_CONCURRENCY, "max_failed_fraction": settings.EXTRACTION_MAX_FAILED_FRACTION,
                   "max_upload_mb": settings.MAX_UPLOAD_MB, "ocr_gpu_fraction": settings.OCR_GPU_MEMORY_UTILIZATION},
    }


@router.get("/generate/runs/{run_id}", response_model=RunStatus, summary="Run state + events since index N")
async def run_status(run_id: str, since: int = 0) -> Dict[str, Any]:
    from api.runs import runs
    run = runs.get(run_id, since)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown run_id.")
    return run


# =====================================================================
# Runtime LLM provider switch (no code edits, no restart)
# =====================================================================

@router.get("/system/providers", summary="Providers, their default models, which one is active, which have a key")
async def list_providers() -> Dict[str, Any]:
    from agents import llm_factory as F
    active = F.resolve_provider()
    return {"active": active, "model": settings.LLM_MODEL or F.DEFAULT_MODELS[active],
            "providers": [{"id": p, "default_model": F.DEFAULT_MODELS[p],
                           "has_key": bool(F.resolve_api_key(p)) or p in ("ollama", "openai_compatible")} for p in F.PROVIDERS]}


@router.post("/system/provider", summary="Test a provider with one tiny call; switch to it only if it answers")
async def switch_provider(body: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio
    from agents import llm_factory as F
    from agents.llm_retry import classify
    p = str(body.get("provider", "")).strip().lower()
    if p not in F.PROVIDERS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown provider {p!r}.")
    if deps.generation_lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT, "A run is in progress; switch after it finishes.")
    old = {k: getattr(settings, k) for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL")}
    settings.LLM_PROVIDER = p
    settings.LLM_MODEL = (body.get("model") or None)
    if body.get("api_key"): settings.LLM_API_KEY = str(body["api_key"])            # memory only, never written to disk
    elif old["LLM_PROVIDER"] != p: settings.LLM_API_KEY = None                    # an override key belongs to the old provider
    if body.get("base_url") is not None: settings.LLM_BASE_URL = body.get("base_url") or None
    t0 = __import__("time").time()
    try:
        llm = F.build_llm("probe")
        reply = await asyncio.wait_for(llm.ainvoke("Reply with exactly: ok"), timeout=45)
    except Exception as exc:
        for k, v in old.items(): setattr(settings, k, v)                         # a failed probe changes nothing
        _, kind = classify(exc)
        return {"ok": False, "provider": old["LLM_PROVIDER"], "error": kind or type(exc).__name__,
                "detail": str(exc)[:300]}
    deps.reset_llm_clients()
    return {"ok": True, "provider": p, "model": settings.LLM_MODEL or F.DEFAULT_MODELS[p],
            "latency_s": round(__import__("time").time() - t0, 2), "reply": str(getattr(reply, "content", ""))[:40]}


def _math_vault(paper_id: str) -> Dict[str, str]:
    """Token -> original LaTeX. Not stored at ingest, but masking is deterministic (counter in document order),
    so re-masking the cached OCR markdown reproduces exactly the same tokens."""
    import json as _j
    from functools import lru_cache
    @lru_cache(maxsize=8)
    def build(pid: str) -> Dict[str, str]:
        f = settings.OCR_CACHE_DIR / f"{pid}.json"
        if not f.exists(): return {}
        md = _j.loads(f.read_text(encoding="utf-8")).get("raw_markdown", "")
        from ingestion.math_masker import MathMasker
        _, v = MathMasker().mask_document(md)
        return {k: b.original_latex for k, b in v.items()}
    global _VAULT_BUILD
    try: _VAULT_BUILD
    except NameError: _VAULT_BUILD = build
    return _VAULT_BUILD(paper_id)
