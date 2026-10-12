import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

def visualize_neo4j_graph(uri="bolt://localhost:7687", user="neo4j", password="password"):
    print("Connecting to Neo4j to build visualization...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    G = nx.DiGraph()
    
    with driver.session() as session:
        # 1. Fetch all nodes
        nodes_result = session.run("MATCH (n) RETURN elementId(n) AS id, n.name AS name, n.entity_type AS type")
        for record in nodes_result:
            # We use the name as the label if it exists, otherwise the elementId
            label = record["name"] if record["name"] else str(record["id"])
            # Shorten the label if it's too long
            if len(label) > 20:
                label = label[:17] + "..."
                
            G.add_node(record["id"], label=label, type=record["type"])
            
        # 2. Fetch all edges
        edges_result = session.run("MATCH (n)-[r]->(m) RETURN elementId(n) AS source, elementId(m) AS target, type(r) AS type")
        for record in edges_result:
            G.add_edge(record["source"], record["target"], label=record["type"])
            
    driver.close()
    
    if len(G.nodes) == 0:
        print("Graph is empty! Did you inject the data yet?")
        return

    print(f"Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    print("Plotting graph...")

    # Plotting setup
    plt.figure(figsize=(12, 8))
    
    # Generate layout
    # spring_layout is usually best for general graphs, but planar is cool if it's a DAG
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Extract labels
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')
    
    plt.title("SoulGraph: Knowledge Graph Visualization", size=15)
    plt.axis('off')
    
    # Save the output
    output_path = Path(__file__).parent / "graph_visualization.png"
    plt.savefig(output_path, format="PNG", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved successfully to: {output_path}")

if __name__ == "__main__":
    visualize_neo4j_graph()
