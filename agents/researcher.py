"""
Researcher Agent (Graph RAG Knowledge Extraction Node)

Queries both Neo4j topological relationships and Vector DB semantic passages
to retrieve all mathematical formulas, hyperparameter defaults, and component lineages
required before architectural generation begins.
"""

from typing import Dict, Any, List
import json

# TODO: Import State schema
from agents.state import SoulGraphState

# TODO: Import Neo4j and Vector Store manager client instances
from db.graph_db import Neo4jGraphManager
from db.vector_db import VectorStoreManager
from config import settings


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
        query_target_component =self.graph_db.query_graph_neighborhood(query_target,depth=2)
        # making the principle vector query
        principle_query=self.vector_db.similarity_search(query_target)

        relevant_formulas = []
        hyperparameters = {}

        for node in query_target_component.get('nodes',[]):
            node_props=node.get('properties',{})
            entity_type = node_props.get('entity_type')

            inner_props=node_props.get('properties','{}')
            inner_props=json.loads(inner_props)

            if entity_type == "Formula":
                # Extract the formula string from the inner properties and add to our list
                if "formula" in inner_props:
                    relevant_formulas.append(inner_props["formula"])

            elif entity_type == "Hyperparameter":
                # Map the hyperparameter name to its value (if it has one)
                name = node_props.get("name", "unknown")
                # We just store whatever is inside its properties (or True if it's just a flag)
                hyperparameters[name] = inner_props

        # 4. Finally, return the partial State dictionary!
        # LangGraph will automatically take these keys and merge them into the master SoulGraphState
        return {
            "paper_context": principle_query,  # The text chunks from VectorDB
            "relevant_formulas": relevant_formulas,
            "hyperparameters": hyperparameters
        }
