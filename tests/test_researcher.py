"""Researcher must turn free text into real graph ids (graph keys are snake_case)."""
import asyncio
import json
from agents.researcher import ResearcherAgent, slugify_node_id


class G:
    def __init__(self, known): self.known, self.asked = known, []
    def search_nodes(self, kw, limit=5): return [{"node_id": "swiglu_activation"}] if "swiglu" in kw.lower() else []
    def query_graph_neighborhood(self, nid, depth=2):
        self.asked.append(nid)
        if nid not in self.known: return {"nodes": [], "edges": []}
        return {"nodes": [{"id": "1", "properties": {"entity_type": "Formula", "name": "f",
                "properties": json.dumps({"formula": "x*sigmoid(x)"})}}], "edges": []}

class V:
    def similarity_search(self, q): return [{"content": "passage"}]


def agent(known):
    a = ResearcherAgent.__new__(ResearcherAgent); a.graph_db, a.vector_db = G(known), V(); return a


def test_slugify():
    assert slugify_node_id("SwiGLU Activation") == "swiglu_activation"
    assert slugify_node_id("  Multi-Head  Attention! ") == "multi_head_attention"


def test_free_text_resolves_to_real_node_and_formulas_flow_through():
    a = agent({"swiglu_activation"})
    out = asyncio.run(a.run({"target_component": "SwiGLU Activation"}))
    assert out["relevant_formulas"] == ["x*sigmoid(x)"]
    assert a.graph_db.asked.count("swiglu_activation") == 1       # slug + search hit are deduped


def test_unknown_component_degrades_to_vector_context_only():
    out = asyncio.run(agent(set()).run({"target_component": "Nonexistent Thing"}))
    assert out["relevant_formulas"] == [] and out["paper_context"] == [{"content": "passage"}]


def test_graph_search_failure_does_not_kill_the_run():
    a = agent({"x"}); a.graph_db.search_nodes = lambda *x, **k: (_ for _ in ()).throw(ConnectionError("down"))
    out = asyncio.run(a.run({"target_component": "x"}))
    assert out["relevant_formulas"] == ["x*sigmoid(x)"]
