"""
Neo4j Graph Database Integration & Cypher Schema Execution

Stores extracted nodes and directed relationships using Cypher MERGE queries to guarantee
idempotent insertions without duplicating theoretical entities across multiple evaluation runs.
"""

from typing import List, Dict, Any, Optional, Tuple
from config import settings
from neo4j import GraphDatabase, Driver, Session, Transaction, exceptions
from neo4j.exceptions import ServiceUnavailable
from ingestion.ontology_extractor import GraphNode, GraphEdge, PaperExtractionSchema
import json

# TODO: Import Neo4j Python Driver and basic exception handlers

# TODO: Import Pydantic ontology schemas


class Neo4jGraphManager:
    """
    Encapsulates lifecycle management, session transactions, and Cypher execution against Neo4j instances.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initializes the Neo4j driver connection pool.
        # TODO: Connect using GraphDatabase.driver(uri, auth=(user, password)).
        # TODO: Store database target name.
        """
        self.uri = uri
        self.auth = (user, password)
        self.database = database
        self.driver = GraphDatabase.driver(
            str(self.uri),
        auth=self.auth)


    def close(self) -> None:
        """
        Closes active driver connection pool cleanly on shutdown.
        # TODO: Call self.driver.close() if active.
        """

        # verify_connectivity() returns None (it raises on failure), so it can't be used as a condition:
        # the old `if` was always False and the driver was never closed.
        driver = getattr(self, "driver", None)
        if driver is not None:
            print('Driver is active and closing it now')
            driver.close()
        else:
            print('Driver is not active or not found')

    def verify_connection(self) -> bool:
        """
        Runs a lightweight ping/verification query against the DBMS server.
        # TODO: Execute simple `RETURN 1 AS test` inside a read transaction.
        # TODO: Return True if successful, log exceptions and return False otherwise.
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    print("Server is running and reachable.")
                    return True
            return False
        except Exception as exc:  # ServiceUnavailable, AuthError, DNS... any failure means "not usable"
            print(f"Server is down or unreachable: {type(exc).__name__}")
            return False

    def create_constraints(self) -> None:
        """
        Creates mandatory Neo4j uniqueness constraints and indexes for high performance lookup.
        # TODO: Execute Cypher commands: `CREATE CONSTRAINT FOR (n:Entity) REQUIRE n.node_id IS UNIQUE`.
        # TODO: Create indices on entity_type and name fields.
        """
        with self.driver.session() as session:
            # 1. Uniqueness Constraint (Automatically indexes node_id)
            session.run("""
                       CREATE CONSTRAINT entity_node_id_unique IF NOT EXISTS
                       FOR (n:Entity)
                       REQUIRE n.node_id IS UNIQUE
                   """)

            # 2. Standard Indexes for performance on non-unique fields
            session.run("""
                        CREATE INDEX entity_type_index IF NOT EXISTS
                            FOR (n:Entity) ON (n.entity_type)
                        """)

            session.run("""
                        CREATE INDEX entity_name_index IF NOT EXISTS
                            FOR (n:Entity) ON (n.name)
                        """)

    def insert_paper_schema(self, schema: Any) -> Dict[str, int]:
        """
        Iterates over a validated PaperExtractionSchema and stores all nodes and edges via Cypher MERGE.
        
        Args:
            schema (PaperExtractionSchema): Validated structure outputted by OntologyExtractor.
            
        Returns:
            Dict[str, int]: Counts of inserted/merged nodes and relationships.
        """
        nodes_inserted = 0
        edges_inserted = 0
        
        with self.driver.session() as session:
            # 1. Insert Nodes
            for node in schema.nodes:
                entity_type_str = node.entity_type.value if hasattr(node.entity_type, 'value') else str(node.entity_type)
                
                # Cypher doesn't allow parameters for Labels (the :Entity part), so we hardcode 'Entity' 
                # but we can set the specific ontology type as a property!
                cypher_node = """
                MERGE (n:Entity {node_id: $node_id})
                SET n.name = $name, 
                    n.description = $desc, 
                    n.entity_type = $type,
                    n.properties = $props
                """
                session.run(
                    cypher_node,
                    node_id=node.node_id,
                    name=node.name,
                    desc=node.description,
                    type=entity_type_str,
                    props=json.dumps(node.properties) if node.properties else "{}"
                )
                nodes_inserted += 1

            # 2. Insert Edges
            for edge in schema.edges:
                rel_type_str = edge.relationship.value if hasattr(edge.relationship, 'value') else str(edge.relationship)
                
                # We use an f-string here ONLY for the relationship type [r:TYPE] because 
                # Neo4j cannot parameterize relationship types dynamically.
                cypher_edge = f"""
                MATCH (src:Entity {{node_id: $src_id}})
                MATCH (tgt:Entity {{node_id: $tgt_id}})
                MERGE (src)-[r:{rel_type_str}]->(tgt)
                SET r.description = $desc
                """
                session.run(
                    cypher_edge,
                    src_id=edge.source_node_id,
                    tgt_id=edge.target_node_id,
                    desc=edge.description or ""
                )
                edges_inserted += 1

        return {"nodes": nodes_inserted, "edges": edges_inserted}

    def query_graph_neighborhood(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Retrieves a local subgraph centered around a specific node.
        
        Args:
            node_id (str): Target entity node_id to expand around.
            depth (int): Hops distance for relationship traversal.
            
        Returns:
            Dict[str, Any]: Subgraph representation consisting of surrounding nodes and linking edges.
        """
        cypher = f"""
        MATCH path = (root:Entity {{node_id: $id}})-[*1..{depth}]-(neighbor:Entity)
        RETURN path
        """
        nodes={}
        edges=[]
        with self.driver.session() as session:
            result = session.run(cypher, id=node_id)

            for record in result:
                path= record["path"]

                for node in path.nodes:
                    nid=node.element_id

                    if nid not in nodes:
                        nodes[nid]={
                            "id":nid,
                            "labels":list(node.labels),
                            "properties":dict(node),

                        }

                for rel in path.relationships:
                    edges.append({
                        "id":rel.element_id,
                        "type":rel.type,
                        "start":rel.start_node.element_id,
                        "end":rel.end_node.element_id,
                        "properties":dict(rel),
                    })
        return{
            'nodes':list(nodes.values()),
            'edges':edges
        }


    def get_full_architecture_lineage(self, paper_title: str) -> List[Dict[str, Any]]:
        """
        Fetches an ordered sequential topological sorting of ArchitectureComponent nodes linked via FEEDS_INTO edges.
        """
        # This matches the longest paths of FEEDS_INTO edges starting from a root node 
        # (a node with no incoming FEEDS_INTO edge)
        cypher = """
        MATCH path = (start:Entity {entity_type: 'ArchitectureComponent'})-[:FEEDS_INTO*]->(end:Entity {entity_type: 'ArchitectureComponent'})
        WHERE NOT ()-[:FEEDS_INTO]->(start)
        RETURN path
        """
        
        lineage_paths = []
        with self.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                path = record["path"]
                # Extract the ordered nodes in this execution flow path
                ordered_nodes = [dict(node) for node in path.nodes]
                lineage_paths.append(ordered_nodes)
                
        return lineage_paths

    def search_nodes(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Case-insensitive lookup of entities by name / node_id (user text -> real graph ids).

        Why: callers hold free text ("SwiGLU Activation") but the graph is keyed by snake_case
        ids ('swiglu_activation'). Exact-id lookups silently return nothing, so resolve first.
        Fully parameterised (no string interpolation of user text into Cypher).
        """
        cypher = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($kw) OR toLower(n.node_id) CONTAINS toLower($kw)
        RETURN n.node_id AS node_id, n.name AS name, n.entity_type AS entity_type
        ORDER BY size(n.node_id) ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(cypher, kw=keyword, limit=int(limit))]
