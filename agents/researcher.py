"""
Researcher Agent (Graph RAG Knowledge Extraction Node)

Queries both Neo4j topological relationships and Vector DB semantic passages
to retrieve all mathematical formulas, hyperparameter defaults, and component lineages
required before architectural generation begins.
"""

from typing import Dict, Any, List
import json
import re

# TODO: Import State schema
from agents.state import SoulGraphState

# TODO: Import Neo4j and Vector Store manager client instances
from db.graph_db import Neo4jGraphManager
from db.vector_db import VectorStoreManager
from config import settings


def slugify_node_id(text: str) -> str:
    """'SwiGLU Activation' -> 'swiglu_activation' (the id convention the extractor enforces)."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


class ResearcherAgent:
    """
    LangGraph node responsible for deep research context gathering from hybrid storage.
    """
    def __init__(self, graph_manager: Any = None, vector_manager: Any = None):
        """
        Initializes references to persistent storage layer managers.
        If none are passed, it creates new default connections.
        """
        self.graph_db = graph_manager or Neo4jGraphManager(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        self.vector_db = vector_manager or VectorStoreManager(
            persistence_dir=str(settings.VECTOR_DB_PATH),
            collection_name=settings.COLLECTION_NAME
        )

    def _neighborhood_for(self, target: str, paper_id: str = None) -> Dict[str, Any]:
        """
        Resolve free text to real graph ids, then merge their depth-2 neighbourhoods.
        Order: exact slug id -> name/id substring search (top 3). Empty graph -> empty result,
        and the vector search still supplies context.
        """
        ids = [slugify_node_id(target)]
        try:
            ids += [h["node_id"] for h in self.graph_db.search_nodes(target, limit=3, paper_id=paper_id)]
        except Exception as exc:  # a graph hiccup must not kill the whole run
            print(f"[Researcher] node search failed: {type(exc).__name__}: {exc}")
        nodes: Dict[str, Any] = {}
        for node_id in dict.fromkeys(i for i in ids if i):
            sub = self.graph_db.query_graph_neighborhood(node_id, depth=2, paper_id=paper_id)
            for n in sub.get("nodes", []):
                nodes.setdefault(n["id"], n)
        return {"nodes": list(nodes.values()), "edges": []}

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph Node Execution Entrypoint for Researcher Agent.
        
        Args:
            state (SoulGraphState): Current state object containing paper_id and target_component.
            
        Returns:
            Dict[str, Any]: Partial dictionary updates applied directly to state (paper_context, relevant_formulas, hyperparameters).
            
        # TODO: Read `target_component` name from state.
        # TODO: Execute Neo4j neighborhood query around target_component node (depth=2) to identify linked Formula and Hyperparameter entities.
        # TODO: Perform Vector DB similarity search using target_component as query to retrieve descriptive text passages.
        # TODO: Aggregate formulas into a clean List[str] and extract hyperparameters into key-value bindings.
        # TODO: Return updated partial state payload to advance workflow to Architect Agent.
        """
        print(f"--- [Researcher Agent] Querying Graph RAG for target: {state.get('target_component')} ---")
        # Extracting the target query from state
        query_target= state.get('target_component')
        # getting the graph NODES FROM OUR Neo4j DB
        paper_id = state.get('paper_id') or None      # scope graph + vectors to the chosen paper
        query_target_component = self._neighborhood_for(query_target or "", paper_id)
        # making the principle vector query
        principle_query=self.vector_db.similarity_search(query_target, filter_metadata={"paper_id": paper_id} if paper_id else None)

        relevant_formulas = []
        hyperparameters = {}

        for node in query_target_component.get('nodes',[]):
            node_props=node.get('properties',{})
            entity_type = node_props.get('entity_type')

            inner_props=node_props.get('properties','{}')
            inner_props=json.loads(inner_props)

            if entity_type == "Formula":
                # Extract the formula string from the inner properties and add to our list
                # The extractor stores LaTeX under "latex" (live graph: 98/125 Formula nodes). Looking only for
                # "formula" meant NO equation ever reached the Coder. Accept both spellings.
                tex = inner_props.get("latex") or inner_props.get("formula")
                if tex:
                    # Prefix with the graph node id so the Coder can cite it as `# Implements: <Node_ID: ...>`
                    # (the Explainer binds code lines back to the paper through those tags).
                    nid = node_props.get("node_id")
                    relevant_formulas.append(f"[{nid}] {tex}" if nid else tex)

            elif entity_type == "Hyperparameter":
                # Map the hyperparameter name to its value (if it has one)
                name = node_props.get("name", "unknown")
                # We just store whatever is inside its properties (or True if it's just a flag)
                # The extractor stores the VALUE in the node description ("h = 8 parallel heads"), not in
                # `properties` (which is usually empty), so pass the description through or no agent ever
                # sees the paper's defaults.
                desc = (node_props.get("description") or "").strip()
                hyperparameters[name] = {**inner_props, **({"description": desc} if desc else {})}

        # 4. Finally, return the partial State dictionary!
        # LangGraph will automatically take these keys and merge them into the master SoulGraphState
        return {
            "paper_context": principle_query,  # The text chunks from VectorDB
            "relevant_formulas": relevant_formulas,
            "hyperparameters": hyperparameters
        }
