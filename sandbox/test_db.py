import sys
import os
from pathlib import Path

# Add project root to path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from db.graph_db import Neo4jGraphManager
from ingestion.ontology_extractor import GraphNode, GraphEdge, PaperExtractionSchema

def test_neo4j():
    print("Testing Neo4j Connection and Injection...")
    
    # 1. Connect to Neo4j
    manager = Neo4jGraphManager(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    print("\n--- Verifying Connection ---")
    manager.verify_connection()
    
    print("\n--- Creating Constraints ---")
    manager.create_constraints()
    print("Constraints created successfully.")
    
    # 2. Create mock schema
    print("\n--- Injecting Mock Graph ---")
    mock_schema = PaperExtractionSchema(
        paper_title="Attention Is All You Need",
        nodes=[
            GraphNode(node_id="transformer", name="Transformer", entity_type="ArchitectureComponent", description="The core attention-based architecture.", properties={"heads": 8}),
            GraphNode(node_id="attention", name="Scaled Dot-Product Attention", entity_type="Concept", description="Computes attention weights.", properties={"formula": "softmax(QK^T/sqrt(d_k))V"}),
            GraphNode(node_id="d_k", name="Dimension of keys", entity_type="Hyperparameter", description="Scaling factor to prevent softmax from vanishing gradients.", properties={})
        ],
        edges=[
            GraphEdge(source_node_id="attention", target_node_id="transformer", relationship="FEEDS_INTO", description="The main building block."),
            GraphEdge(source_node_id="d_k", target_node_id="attention", relationship="PARAMETERIZES", description="Used as the scaling denominator in the softmax.")
        ]
    )
    
    stats = manager.insert_paper_schema(mock_schema)
    print(f"Injection Complete: {stats['nodes']} nodes and {stats['edges']} edges merged.")
    
    print("\n--- Querying Neighborhood of 'attention' ---")
    neighborhood = manager.query_graph_neighborhood("attention", depth=1)
    
    print("Neighborhood Nodes:")
    for n in neighborhood['nodes']:
        print(f"  - {n['properties']['name']} ({n['labels'][0]})")
        
    print("Neighborhood Edges:")
    for e in neighborhood['edges']:
        print(f"  - {e['start']} -[{e['type']}]-> {e['end']}")
        
    manager.close()
    
if __name__ == "__main__":
    test_neo4j()
