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
import re

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
            # Identity is (node_id, paper_id): a global node_id constraint would make two papers that both
            # define 'multi_head_attention' collide. Drop the legacy global one if an old DB has it.
            session.run("DROP CONSTRAINT entity_node_id_unique IF EXISTS")
            session.run("""
                       CREATE CONSTRAINT entity_node_paper_unique IF NOT EXISTS
                       FOR (n:Entity)
                       REQUIRE (n.node_id, n.paper_id) IS UNIQUE
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

    def query_graph_neighborhood(self, node_id: str, depth: int = 2, paper_id: Optional[str] = None) -> Dict[str, Any]:
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
        WHERE $pid IS NULL OR root.paper_id = $pid
        RETURN path
        """
        nodes={}
        edges=[]
        with self.driver.session() as session:
            result = session.run(cypher, id=node_id, pid=paper_id)

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

    def search_nodes(self, keyword: str, limit: int = 5, paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Case-insensitive lookup of entities by name / node_id (user text -> real graph ids).

        Why: callers hold free text ("SwiGLU Activation") but the graph is keyed by snake_case
        ids ('swiglu_activation'). Exact-id lookups silently return nothing, so resolve first.
        Fully parameterised (no string interpolation of user text into Cypher).
        """
        cypher = """
        MATCH (n:Entity)
        WHERE (toLower(n.name) CONTAINS toLower($kw) OR toLower(n.node_id) CONTAINS toLower($kw))
          AND ($pid IS NULL OR n.paper_id = $pid)
        RETURN n.node_id AS node_id, n.name AS name, n.entity_type AS entity_type
        ORDER BY size(n.node_id) ASC
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(cypher, kw=keyword, limit=int(limit), pid=paper_id)]


    _REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")

    def replace_paper_schema(self, schema: Any, paper_id: str, title: Optional[str] = None) -> Dict[str, int]:
        """
        Idempotent, atomic (re)write of ONE paper's subgraph.

        Why: LLM extraction is non-deterministic, so re-ingesting a paper yields different node ids
        ('multi_head_attention' vs 'multi_headed_attention'); plain MERGE then piles a second near-copy
        on top of the first (measured: 987 -> 1701 nodes on a re-run). Replace semantics fix that:
        delete everything tagged with this paper_id, then write the new graph, in ONE transaction so a
        failure midway rolls back and the old graph survives.
        """
        node_rows: Dict[str, Dict[str, Any]] = {}
        for n in schema.nodes:
            node_rows.setdefault(n.node_id, {
                "node_id": n.node_id, "name": n.name, "desc": n.description,
                "type": n.entity_type.value if hasattr(n.entity_type, "value") else str(n.entity_type),
                "props": json.dumps(n.properties) if n.properties else "{}",
            })

        by_type: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
        for e in schema.edges:
            rel = e.relationship.value if hasattr(e.relationship, "value") else str(e.relationship)
            if not self._REL_TYPE.match(rel):   # the type is interpolated into Cypher: whitelist its shape
                raise ValueError(f"unsafe relationship type {rel!r}")
            by_type.setdefault(rel, {})[(e.source_node_id, e.target_node_id)] = {
                "src": e.source_node_id, "tgt": e.target_node_id, "desc": e.description or ""}

        def work(tx):
            tx.run("MATCH (n:Entity {paper_id: $pid}) DETACH DELETE n", pid=paper_id).consume()
            c = tx.run(
                """UNWIND $rows AS r
                   MERGE (n:Entity {node_id: r.node_id, paper_id: $pid})
                   SET n.name = r.name, n.description = r.desc, n.entity_type = r.type, n.properties = r.props,
                       n.paper_title = $title""",
                rows=list(node_rows.values()), pid=paper_id, title=title).consume().counters.nodes_created
            rels = 0
            for rel, rows in by_type.items():
                rels += tx.run(
                    f"""UNWIND $rows AS r
                        MATCH (a:Entity {{node_id: r.src, paper_id: $pid}})
                        MATCH (b:Entity {{node_id: r.tgt, paper_id: $pid}})
                        MERGE (a)-[x:{rel}]->(b)
                        SET x.description = r.desc""",
                    rows=list(rows.values()), pid=paper_id).consume().counters.relationships_created
            return {"nodes": c, "edges": rels}

        with self.driver.session(database=self.database) as session:
            return session.execute_write(work)

    def delete_paper(self, paper_id: str) -> int:
        """Remove one paper's whole subgraph. Returns nodes deleted."""
        with self.driver.session(database=self.database) as session:
            return session.run("MATCH (n:Entity {paper_id: $pid}) DETACH DELETE n", pid=paper_id).consume().counters.nodes_deleted


    # ------------------------------------------------------------------ read models for the UI
    def list_papers(self) -> List[Dict[str, Any]]:
        """One row per ingested paper: id, best-guess title, node/edge counts. The title is the `Paper`
        entity with the most connections (the extractor emits many Paper nodes, one per cited work) only as a
        fallback: ingest stores the voted title in `paper_title`."""
        cypher = """
        MATCH (n:Entity) WHERE n.paper_id IS NOT NULL
        OPTIONAL MATCH (n)-[r]-()
        WITH n.paper_id AS pid, n, count(r) AS deg
        RETURN pid, count(n) AS nodes, sum(deg) / 2 AS edges, max(n.paper_title) AS stored_title,
               collect({name: n.name, t: n.entity_type, deg: deg}) AS ns
        """
        out = []
        with self.driver.session() as session:
            for r in session.run(cypher):
                papers = [x for x in r["ns"] if x["t"] == "Paper" and x["name"]]
                title = r["stored_title"] or (max(papers, key=lambda x: x["deg"])["name"] if papers else r["pid"])
                out.append({"paper_id": r["pid"], "title": title, "nodes": r["nodes"], "edges": int(r["edges"])})
        return sorted(out, key=lambda p: -p["nodes"])

    def paper_overview(self, paper_id: str, limit: int = 120, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """The `limit` best-connected entities of a paper plus the edges among them (a drawable skeleton,
        not the whole hairball: a typical paper has ~1000 nodes). `include` force-adds specific node ids: a run
        cites low-degree formula nodes that would never make the top-N, and the UI must be able to show them."""
        limit = max(5, min(int(limit), 400))
        node_q = """
        MATCH (n:Entity {paper_id: $pid})
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS deg ORDER BY deg DESC LIMIT $limit
        RETURN n.node_id AS id, n.name AS name, n.entity_type AS type, n.description AS description, deg AS degree
        """
        extra_q = """
        MATCH (n:Entity {paper_id: $pid}) WHERE n.node_id IN $ids
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS deg
        RETURN n.node_id AS id, n.name AS name, n.entity_type AS type, n.description AS description, deg AS degree
        """
        edge_q = """
        MATCH (a:Entity {paper_id: $pid})-[r]->(b:Entity {paper_id: $pid})
        WHERE a.node_id IN $ids AND b.node_id IN $ids
        RETURN a.node_id AS source, b.node_id AS target, type(r) AS type
        """
        with self.driver.session() as session:
            nodes = [dict(r) for r in session.run(node_q, pid=paper_id, limit=limit)]
            have = {n["id"] for n in nodes}
            extra = [i for i in dict.fromkeys(include or []) if i not in have][:60]
            if extra:
                nodes += [dict(r) for r in session.run(extra_q, pid=paper_id, ids=extra)]
            ids = [n["id"] for n in nodes]
            edges = [dict(r) for r in session.run(edge_q, pid=paper_id, ids=ids)] if ids else []
        return {"nodes": nodes, "edges": edges}

    def top_components(self, paper_id: str, limit: int = 24) -> List[Dict[str, Any]]:
        """Candidate targets for code generation, best first. Ranked by how many Formula entities sit within
        2 hops (those become `# Implements:` tags in the code and light up the graph), then by connectivity.
        Candidates are ArchitectureComponents plus any entity with formulas nearby (e.g. 'Multi-Head Attention'
        is extracted as a Concept but is exactly what a user wants to generate)."""
        cypher = """
        MATCH (n:Entity {paper_id: $pid})
        WHERE n.entity_type <> 'Paper' AND n.entity_type <> 'Formula'
        OPTIONAL MATCH (n)-[*1..2]-(f:Entity {paper_id: $pid, entity_type: 'Formula'})
        WITH n, count(DISTINCT f) AS formulas
        OPTIONAL MATCH (n)-[r]-()
        WITH n, formulas, count(r) AS deg
        WHERE n.entity_type = 'ArchitectureComponent' OR formulas > 0
        RETURN n.node_id AS id, n.name AS name, n.description AS description, deg AS degree, formulas
        ORDER BY (formulas > 0) DESC, deg DESC LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(cypher, pid=paper_id, limit=max(1, min(int(limit), 100)))]
