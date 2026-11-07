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
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from api import deps
from api.jobs import jobs
from api.schemas import (
    AblationStudyRequest, AblationStudyResponse,
    CodeGenerationRequest, CodeGenerationResponse,
    ExplainerChatRequest, ExplainerChatResponse,
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
    return AblationStudyResponse(
        mutation_prompt=request.mutation_prompt,
        baseline_code=request.baseline_code,
        ablated_code=report["mutated_code"],
        baseline_metrics=report["baseline_metrics"],
        ablated_metrics=report["ablated_metrics"],
        diff_explanation=report["diff_summary"],
    )


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
