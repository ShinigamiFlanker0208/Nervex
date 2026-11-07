"""
Lazy, cached heavy dependencies (Neo4j driver, Chroma, Marker OCR models, LLM agents).

Why: importing the API must not connect to Neo4j or load GBs of OCR weights. Everything is built on
first use and cached, and routes fetch it through these functions so tests can monkeypatch them.
`generation_lock` serialises GPU-bound work: one RTX-class GPU cannot run two sandboxes + OCR safely.
"""
import asyncio
import threading
from functools import lru_cache
from typing import Any

from config import settings

generation_lock = asyncio.Semaphore(1)       # /generate/code, ablation
ingestion_lock = threading.Lock()            # /ingest/pdf background worker


@lru_cache(maxsize=1)
def get_graph_db() -> Any:
    from db.graph_db import Neo4jGraphManager
    return Neo4jGraphManager(
        uri=settings.NEO4J_URI, user=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD, database=settings.NEO4J_DATABASE,
    )


@lru_cache(maxsize=1)
def get_vector_db() -> Any:
    from db.vector_db import VectorStoreManager
    return VectorStoreManager(
        persistence_dir=str(settings.VECTOR_DB_PATH), collection_name=settings.COLLECTION_NAME
    )


@lru_cache(maxsize=1)
def get_pdf_parser() -> Any:
    # Isolated child process per document: see ingestion/ocr_worker.py for why (VRAM lifetime).
    from ingestion.ocr_worker import IsolatedPDFParser
    return IsolatedPDFParser(use_cuda=settings.ENABLE_GPU_PASSTHRU, device_id=settings.CUDA_DEVICE_ID)


def get_masker() -> Any:
    from ingestion.math_masker import MathMasker
    return MathMasker()  # stateful vault -> fresh instance per document


def get_chunker() -> Any:
    from ingestion.semantic_chunker import SemanticChunker
    return SemanticChunker(chunk_size=1000, chunk_overlap=150)


@lru_cache(maxsize=1)
def get_extractor() -> Any:
    from ingestion.ontology_extractor import OntologyExtractor
    return OntologyExtractor()


def get_orchestrator() -> Any:
    from agents.graph import SoulGraphOrchestrator
    orch = SoulGraphOrchestrator(graph_manager=get_graph_db(), vector_manager=get_vector_db())
    orch.build_graph()
    return orch


@lru_cache(maxsize=1)
def get_transpiler() -> Any:
    from features.transpilation import FrameworkTranspiler
    return FrameworkTranspiler()


@lru_cache(maxsize=1)
def get_explainer() -> Any:
    from explainer.chat_agent import ExplainerChatAgent
    return ExplainerChatAgent()
