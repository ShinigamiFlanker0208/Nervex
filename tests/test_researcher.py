"""Researcher must turn free text into real graph ids (graph keys are snake_case)."""
import asyncio
import json
from agents.researcher import ResearcherAgent, slugify_node_id


class G:
    def __init__(self, known): self.known, self.asked = known, []
    def search_nodes(self, kw, limit=5, paper_id=None): return [{"node_id": "swiglu_activation"}] if "swiglu" in kw.lower() else []
    def query_graph_neighborhood(self, nid, depth=2, paper_id=None):
        self.asked.append(nid)
        if nid not in self.known: return {"nodes": [], "edges": []}
        return {"nodes": [{"id": "1", "properties": {"entity_type": "Formula", "name": "f", "node_id": "swiglu_formula",
                "properties": json.dumps({"formula": "x*sigmoid(x)"})}}], "edges": []}

class V:
    def similarity_search(self, q, filter_metadata=None): return [{"content": "passage"}]


def agent(known):
    a = ResearcherAgent.__new__(ResearcherAgent); a.graph_db, a.vector_db = G(known), V(); return a


def test_slugify():
    assert slugify_node_id("SwiGLU Activation") == "swiglu_activation"
    assert slugify_node_id("  Multi-Head  Attention! ") == "multi_head_attention"


def test_free_text_resolves_to_real_node_and_formulas_flow_through():
    a = agent({"swiglu_activation"})
    out = asyncio.run(a.run({"target_component": "SwiGLU Activation"}))
    assert out["relevant_formulas"] == ["[swiglu_formula] x*sigmoid(x)"]
    assert a.graph_db.asked.count("swiglu_activation") == 1       # slug + search hit are deduped


def test_unknown_component_degrades_to_vector_context_only():
    out = asyncio.run(agent(set()).run({"target_component": "Nonexistent Thing"}))
    assert out["relevant_formulas"] == [] and out["paper_context"] == [{"content": "passage"}]


def test_graph_search_failure_does_not_kill_the_run():
    a = agent({"x"}); a.graph_db.search_nodes = lambda *x, **k: (_ for _ in ()).throw(ConnectionError("down"))
    out = asyncio.run(a.run({"target_component": "x"}))
    assert out["relevant_formulas"] == ["[swiglu_formula] x*sigmoid(x)"]


def test_graph_manager_verify_and_close_contract():
    """Regression: verify_connection used to return None always; close() never closed the driver."""
    from db.graph_db import Neo4jGraphManager

    class Sess:
        def __init__(self, ok): self.ok = ok
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, q):
            if not self.ok: raise ConnectionError("down")
            class R:
                def single(self_): return {"test": 1}
            return R()

    class Drv:
        closed = False
        def __init__(self, ok): self.ok = ok
        def session(self): return Sess(self.ok)
        def close(self): Drv.closed = True

    m = Neo4jGraphManager.__new__(Neo4jGraphManager)
    m.driver = Drv(True);  assert m.verify_connection() is True
    m.driver = Drv(False); assert m.verify_connection() is False
    m.driver = Drv(True);  m.close(); assert Drv.closed is True


def test_hyperparameter_description_reaches_agents():
    """Live E2E: graph stores hyperparameters as names only ('h': {}); the value is in the description."""
    class G2(G):
        def query_graph_neighborhood(self, nid, depth=2, paper_id=None):
            return {"nodes": [{"id": "9", "properties": {"entity_type": "Hyperparameter", "name": "h",
                    "description": "number of parallel attention heads, h = 8", "properties": "{}"}}], "edges": []}
    a = ResearcherAgent.__new__(ResearcherAgent); a.graph_db, a.vector_db = G2(set()), V()
    out = asyncio.run(a.run({"target_component": "x"}))
    assert "h = 8" in out["hyperparameters"]["h"]["description"]
