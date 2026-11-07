"""Ablation, transpiler, explainer loop with fake LLMs/orchestrators."""
import asyncio
import pytest
from types import SimpleNamespace

from features.ablation import AblationStudyController
from features.transpilation import FrameworkTranspiler
from explainer.chat_agent import ExplainerChatAgent


def run(c): return asyncio.run(c)


# ---------------- ablation ----------------
class Orch:
    def __init__(self): self.got = None
    async def run_translation_loop(self, st):
        self.got = st
        return {**st, "current_code": "y = x", "execution_status": "SUCCESS", "attempt_count": 1,
                "tensor_shape_traces": {"fc": {}}}

BASE = {"current_code": "y = fc(x) + x", "execution_status": "SUCCESS", "attempt_count": 3,
        "execution_logs": [{"t": 1}], "tensor_shape_traces": {"fc": {}, "norm": {}}}


def test_ablation_resets_state_and_injects_mutation_where_coder_reads_it():
    o = Orch()
    rep = run(AblationStudyController(o, None).run_ablation_experiment(BASE, "remove residual"))
    assert o.got["attempt_count"] == 0 and o.got["execution_logs"] == []
    assert "remove residual" in o.got["critic_feedback"]
    assert o.got["current_code"] == BASE["current_code"]          # coder sees code to modify
    assert BASE["attempt_count"] == 3                              # baseline not mutated (deepcopy)
    assert rep["ablated_metrics"]["traced_layers"] == 1 and rep["baseline_metrics"]["traced_layers"] == 2
    assert "norm" in rep["diff_summary"]                           # layer removed is reported


def test_ablation_refuses_failed_baseline_and_empty_instruction():
    c = AblationStudyController(Orch(), None)
    with pytest.raises(ValueError): run(c.run_ablation_experiment({**BASE, "execution_status": "RUNTIME_ERROR"}, "x"))
    with pytest.raises(ValueError): run(c.run_ablation_experiment(BASE, "  "))


# ---------------- transpiler ----------------
class LLM:
    def __init__(self, *replies): self.replies, self.calls = list(replies), 0
    async def ainvoke(self, msgs):
        self.calls += 1
        return SimpleNamespace(content=self.replies.pop(0), tool_calls=[])

class Critic:
    def run_static_analysis(self, code): return ("import os" not in code, "bad import os")


def test_transpile_same_framework_returns_early_without_llm():
    llm = LLM()
    out = run(FrameworkTranspiler(llm=llm, critic=Critic()).transpile_model("c", "PyTorch", "pytorch", ""))
    assert out["transpiled_code"] == "c" and llm.calls == 0


def test_transpile_ok_normalises_names_and_splits_notes():
    llm = LLM("```python\nimport jax\n```\n### Notes\n- keys split")
    out = run(FrameworkTranspiler(llm=llm, critic=Critic()).transpile_model("c", "PyTorch", "JAX/Equinox", "sk"))
    assert out["target_framework"] == "JAX" and out["syntax_valid"] and "keys split" in out["notes"]


def test_transpile_repairs_after_static_failure():
    llm = LLM("```python\nimport os\n```", "```python\nimport jax\n```")
    out = run(FrameworkTranspiler(llm=llm, critic=Critic()).transpile_model("c", "PyTorch", "JAX", ""))
    assert llm.calls == 2 and out["syntax_valid"]


def test_transpile_reports_failure_after_all_bad_attempts_and_missing_block():
    llm = LLM("```python\nimport os\n```", "no code here", "```python\nimport os\n```", "```python\nimport os\n```")
    out = run(FrameworkTranspiler(llm=llm, critic=Critic()).transpile_model("c", "PyTorch", "JAX", ""))
    assert not out["syntax_valid"] and "STATIC ANALYSIS FAILED" in out["notes"]


def test_transpile_unknown_framework():
    with pytest.raises(ValueError):
        run(FrameworkTranspiler(llm=LLM(), critic=Critic()).transpile_model("c", "tensorflow", "JAX", ""))


# ---------------- explainer ----------------
class ToolLLM:
    """First reply asks for a tool, second answers."""
    def __init__(self): self.n = 0
    def bind_tools(self, tools): return self
    async def ainvoke(self, msgs):
        self.n += 1
        if self.n == 1:
            return SimpleNamespace(content="", tool_calls=[{"name": "search_paper_text", "args": {"query_string": "gelu"}, "id": "1"}])
        return SimpleNamespace(content="Because GELU.", tool_calls=[])

SCRIPT = "def f(x):\n    # Implements: <Node_ID: Eq_9>\n    # [B, D] -> [B, D]\n    y = x\n    return y\n"


def test_explainer_runs_tool_and_collects_citations(monkeypatch):
    import explainer.chat_agent as ca
    monkeypatch.setattr(ca, "search_paper_text", lambda query_string, n_results=3: ["authors prefer GELU"])
    agent = ExplainerChatAgent(llm=ToolLLM())
    agent._tools["search_paper_text"] = ca.search_paper_text
    out = run(agent.explain("why?", (4, 4), SCRIPT, []))
    assert out["answer_markdown"] == "Because GELU."
    assert out["cited_graph_nodes"] == ["Eq_9"]
    assert out["cited_paper_passages"] == ["authors prefer GELU"]
    assert out["inferred_tensor_shapes"]


def test_explainer_survives_unparseable_script_and_tool_errors():
    class L(ToolLLM):
        pass
    agent = ExplainerChatAgent(llm=L())
    agent._tools["search_paper_text"] = lambda **k: (_ for _ in ()).throw(RuntimeError("chroma down"))
    out = run(agent.explain("why?", (1, 1), "def (:", []))
    assert out["answer_markdown"] == "Because GELU."      # tool error is fed back to the model, no crash


def test_explainer_stops_on_endless_tool_calls():
    class Loop:
        def bind_tools(self, t): return self
        async def ainvoke(self, m):
            return SimpleNamespace(content="", tool_calls=[{"name": "search_paper_text", "args": {"query_string": "q"}, "id": "x"}])
    agent = ExplainerChatAgent(llm=Loop())
    agent._tools["search_paper_text"] = lambda **k: ["p"]
    out = run(agent.explain("why?", (1, 1), "x=1", []))
    assert "budget exhausted" in out["answer_markdown"]


# ---------------- transpile: runtime verification + repair ----------------
def test_transpile_runtime_failure_is_fed_back_and_repaired():
    llm = LLM("```python\nbad_runtime = 1\n```", "```python\nok = 1\n```\n### Notes\n- fixed")
    seen = []

    async def verifier(code, fw, inputs):
        seen.append((code, fw))
        return ("ok" in code), "Traceback: vmap exploded", {"output": [2, 3]}
    out = run(FrameworkTranspiler(llm=llm, critic=Critic(), verifier=verifier)
              .transpile_model("c", "PyTorch", "JAX", "", {"x": "[2, 3]"}))
    assert out["runtime_verified"] is True and out["output_shape"] == [2, 3] and llm.calls == 2
    assert [fw for _, fw in seen] == ["JAX", "JAX"]


def test_transpile_reports_runtime_failure_honestly():
    llm = LLM(*["```python\nx = 1\n```"] * 4)

    async def verifier(code, fw, inputs):
        return False, "boom", {}
    out = run(FrameworkTranspiler(llm=llm, critic=Critic(), verifier=verifier)
              .transpile_model("c", "PyTorch", "JAX", "", {"x": "[2, 3]"}))
    assert out["syntax_valid"] is True and out["runtime_verified"] is False and "RUNTIME FAILED" in out["notes"]


def test_transpile_without_inputs_says_not_executed():
    async def verifier(*a):
        raise AssertionError("must not run")
    out = run(FrameworkTranspiler(llm=LLM("```python\nx = 1\n```"), critic=Critic(), verifier=verifier)
              .transpile_model("c", "PyTorch", "JAX", ""))
    assert out["runtime_verified"] is None and "NOT EXECUTED" in out["notes"]
