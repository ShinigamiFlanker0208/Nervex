"""
API Request & Response Data Transmission Schemas (Pydantic)

Defines strict JSON typing contracts between Frontend UI and Backend processing routers.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# =====================================================================
# Ingestion & Graph RAG Schemas
# =====================================================================

class IngestionResponse(BaseModel):
    paper_id: str = Field(..., description="Unique deterministic identifier for processed paper")
    paper_title: str
    num_pages: int
    num_entities_extracted: int
    num_relationships_merged: int
    status_message: str


class KnowledgeGraphQueryRequest(BaseModel):
    node_id: Optional[str] = Field(None, description="Specific node ID to query neighborhood")
    search_keyword: Optional[str] = Field(None, description="General search term across nodes")
    depth: int = Field(2, description="Hop depth for neighborhood expansion")


class GraphNeighborhoodResponse(BaseModel):
    center_node_id: Optional[str]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# =====================================================================
# Code Generation & Sandbox Execution Schemas
# =====================================================================

class CodeGenerationRequest(BaseModel):
    paper_id: str = Field(..., description="Target paper ID previously ingested")
    target_component: str = Field(..., description="Architecture block name (e.g., 'TransformerEncoder')")
    target_framework: str = Field("PyTorch", description="Requested backend framework ('PyTorch' or 'JAX')")


class CodeGenerationResponse(BaseModel):
    paper_id: str
    target_component: str
    target_framework: str
    generated_code: str
    attempt_count: int
    execution_status: str
    tensor_shape_traces: Dict[str, Any] = Field(default_factory=dict, description="Captured layer telemetry tooltips")
    error_logs: List[Dict[str, Any]] = Field(default_factory=list)


# =====================================================================
# Explainer UI Chat Schemas
# =====================================================================

class ExplainerChatRequest(BaseModel):
    paper_id: str
    user_question: str
    highlighted_line_start: int
    highlighted_line_end: int
    current_code_script: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class ExplainerChatResponse(BaseModel):
    answer_markdown: str
    cited_graph_nodes: List[str]
    cited_paper_passages: List[str]
    inferred_tensor_shapes: Optional[Dict[str, str]]


# =====================================================================
# Advanced Features Schemas (Ablation & Transpilation)
# =====================================================================

class AblationStudyRequest(BaseModel):
    paper_id: str
    target_component: str
    baseline_code: str
    mutation_prompt: str = Field(..., description="Natural language mutation instruction")


class AblationStudyResponse(BaseModel):
    mutation_prompt: str
    baseline_code: str
    ablated_code: str
    baseline_metrics: Dict[str, Any]
    ablated_metrics: Dict[str, Any]
    diff_explanation: str


class TranspilationRequest(BaseModel):
    source_code: str
    source_framework: str
    target_framework: str
    architecture_skeleton: str


class TranspilationResponse(BaseModel):
    source_framework: str
    target_framework: str
    transpiled_code: str
    syntax_valid: bool
    notes: str
