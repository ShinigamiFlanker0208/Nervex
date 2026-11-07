"""
REAL Neo4j integration tests for per-paper identity. Skipped automatically when Neo4j is not reachable
(CI has none), so the unit suite stays hermetic. Run locally with the container from the README.

Why real DB: the bug these guard (987 -> 1701 nodes on re-ingest) lives in Cypher MERGE semantics and
transaction behaviour, which a fake cannot reproduce.
"""
import uuid
from types import SimpleNamespace as NS

import pytest

from config import settings
from db.graph_db import Neo4jGraphManager


def _node(i, t="Concept"): return NS(node_id=i, name=i, description=f"d {i}", entity_type=NS(value=t), properties={})
def _edge(a, b, r="FEEDS_INTO"): return NS(source_node_id=a, target_node_id=b, relationship=NS(value=r), description="")
def _schema(ids, edges=()): return NS(nodes=[_node(i) for i in ids], edges=[_edge(*e) for e in edges])


@pytest.fixture(scope="module")
def g():
    try:
        mgr = Neo4jGraphManager(settings.NEO4J_URI, settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        mgr.driver.verify_connectivity()
    except Exception:
        pytest.skip("Neo4j not reachable")
    yield mgr
    mgr.close()


@pytest.fixture
def pid(g):
    p = "test_" + uuid.uuid4().hex[:8]
    yield p
    g.delete_paper(p)


def count(g, pid):
    with g.driver.session() as s:
        return s.run("MATCH (n:Entity {paper_id:$p}) RETURN count(n) AS c", p=pid).single()["c"]


def test_replace_is_idempotent_even_when_llm_renames_nodes(g, pid):
    g.replace_paper_schema(_schema(["multi_head_attention", "query"], [("query", "multi_head_attention")]), pid)
    assert count(g, pid) == 2
    # second run: the LLM names things differently -> old nodes must NOT linger
    out = g.replace_paper_schema(_schema(["multi_headed_attention", "query", "key"], [("query", "key")]), pid)
    assert count(g, pid) == 3 and out == {"nodes": 3, "edges": 1}
    with g.driver.session() as s:
        assert s.run("MATCH (n:Entity {paper_id:$p, node_id:'multi_head_attention'}) RETURN n", p=pid).single() is None


def test_same_node_id_in_two_papers_does_not_collide(g, pid):
    other = pid + "_b"
    try:
        g.replace_paper_schema(_schema(["x", "y"], [("x", "y")]), pid)
        g.replace_paper_schema(_schema(["x"]), other)
        assert count(g, pid) == 2 and count(g, other) == 1
        a = g.query_graph_neighborhood("x", 1, paper_id=pid)
        b = g.query_graph_neighborhood("x", 1, paper_id=other)
        assert len(a["nodes"]) == 2 and b["nodes"] == []            # scoped: B's x has no neighbours
        assert {h["node_id"] for h in g.search_nodes("x", paper_id=other)} == {"x"}
    finally:
        g.delete_paper(other)


def test_failed_replace_rolls_back_and_keeps_the_old_graph(g, pid):
    g.replace_paper_schema(_schema(["keep_me"]), pid)
    bad = _schema(["a"], [("a", "a", "bad type; MATCH (n) DETACH DELETE n //")])
    with pytest.raises(ValueError):
        g.replace_paper_schema(bad, pid)                              # rejected before touching the DB
    assert count(g, pid) == 1


def test_ghost_edges_are_not_counted_or_created(g, pid):
    out = g.replace_paper_schema(_schema(["a"], [("a", "missing")]), pid)
    assert out["edges"] == 0


def test_unscoped_query_keeps_legacy_behaviour(g, pid):
    g.replace_paper_schema(_schema(["legacy_a", "legacy_b"], [("legacy_a", "legacy_b")]), pid)
    assert len(g.query_graph_neighborhood("legacy_a", 1)["nodes"]) == 2


def test_constraints_allow_same_id_across_papers(g, pid):
    g.create_constraints()
    other = pid + "_c"
    try:
        g.replace_paper_schema(_schema(["dup"]), pid)
        g.replace_paper_schema(_schema(["dup"]), other)               # would raise under the old global constraint
        assert count(g, other) == 1
    finally:
        g.delete_paper(other)
