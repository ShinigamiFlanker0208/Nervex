"""
SoulGraph Entrypoint & Server Configuration

Initializes the FastAPI application, mounts middlewares, connects routers,
and handles application lifecycle events (e.g., verifying database connectivity).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import logging

from config import settings
from api.routes import router as api_router
from api import deps

log = logging.getLogger("nervex")


async def _probe_subsystems() -> Dict[str, Any]:
    """
    Live subsystem status. Each probe is isolated + time-boxed: a dead Neo4j must make /health say
    "degraded", never hang or crash the server (and never block the event loop).
    """
    def neo4j_ok() -> bool:
        return bool(deps.get_graph_db().verify_connection())

    def docker_ok() -> bool:
        import docker
        return bool(docker.from_env().ping())

    def sandbox_image_ok() -> bool:
        import docker
        docker.from_env().images.get(settings.DOCKER_SANDBOX_IMAGE)
        return True

    def cuda_ok() -> bool:
        import torch
        return bool(torch.cuda.is_available())

    checks = {"neo4j": neo4j_ok, "docker": docker_ok, "sandbox_image": sandbox_image_ok, "cuda": cuda_ok}
    out: Dict[str, Any] = {}
    for name, fn in checks.items():
        try:
            out[name] = await asyncio.wait_for(asyncio.to_thread(fn), timeout=5)
        except Exception as exc:
            out[name] = False
            out[f"{name}_error"] = f"{type(exc).__name__}"
    return out


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan event handler for FastAPI startup and shutdown.
    # TODO: Implement startup verification (Neo4j ping, Docker daemon verification, local model download check).
    # TODO: Implement graceful shutdown (close Neo4j sessions, cleanup dangling Docker sandboxes).
    """
    print("--- [SoulGraph] Initiating system startup & readiness checks ---")
    settings.create_directories()
    status_ = await _probe_subsystems()
    for name in ("neo4j", "docker", "sandbox_image", "cuda"):
        if not status_.get(name):
            # Warn, don't die: the API (e.g. /health, /explainer with a cached graph) is still useful.
            log.warning("startup check FAILED: %s %s", name, status_.get(f"{name}_error", ""))
    if not status_.get("sandbox_image"):
        log.warning("Build the sandbox: docker build -f sandbox/Dockerfile.sandbox -t %s .", settings.DOCKER_SANDBOX_IMAGE)

    yield

    print("--- [SoulGraph] Shutting down resources ---")
    if deps.get_graph_db.cache_info().currsize:   # only close it if it was ever opened
        try:
            deps.get_graph_db().close()
        except Exception:
            log.exception("error closing Neo4j driver")


app = FastAPI(
    title="SoulGraph Engine",
    description="Autonomous Machine Learning Research Translation & Execution Engine",
    version="0.1.0",
    lifespan=lifespan
)

# Explicit origins (a "*" origin with credentials is rejected by browsers anyway). Edit CORS_ORIGINS in .env.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", summary="Root Health Endpoint")
async def root_health_check() -> Dict[str, Any]:
    """Live status of every subsystem. 200 always (it is a status report); `status` says online/degraded."""
    probes = await _probe_subsystems()
    required = ("neo4j", "docker", "sandbox_image")
    ok = all(probes.get(k) for k in required)
    return {
        "status": "online" if ok else "degraded",
        "system": "SoulGraph Autonomous Engine",
        "subsystems": probes,
        "message": "all required subsystems reachable" if ok else
                   "degraded: " + ", ".join(k for k in required if not probes.get(k)),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
