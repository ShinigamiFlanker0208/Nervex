"""
Ablation Study Mode Mutation Engine

Processes user ablation prompts (e.g., "Remove residual connections and test again"), mutates
the LangGraph architectural structural skeleton and Knowledge Graph representation, re-runs
autonomous coding & sandboxed execution, and outputs comparative execution metrics and tensor behavior logs.
"""

from typing import Dict, Any, List, Optional, Tuple
import copy
import difflib

# TODO: Import LangGraph orchestrator and database managers
# from agents.graph import SoulGraphOrchestrator
# from db.graph_db import Neo4jGraphManager
# from sandbox.docker_controller import DockerSandboxController


class AblationStudyController:
    """
    Orchestrates comparative model ablation studies and architectural mutation runs.
    """

    def __init__(self, orchestrator: Any, sandbox_controller: Any):
        """
        Initializes references to workflow orchestrators and sandbox environments.
        # TODO: Store orchestrator and sandbox_controller instances.
        """
        self.orchestrator = orchestrator
        self.sandbox = sandbox_controller

    async def run_ablation_experiment(self, baseline_state: Dict[str, Any], mutation_instruction: str) -> Dict[str, Any]:
        """
        Executes an architectural mutation against a baseline working model and captures comparative results.
        
        Args:
            baseline_state (Dict[str, Any]): Terminal success state of the unmodified model generation run.
            mutation_instruction (str): Natural language modification request (e.g., "Remove the LayerNorm before MultiHeadAttention").
            
        Returns:
            Dict[str, Any]: Comparative report containing:
                - "baseline_metrics": Peak memory, execution latency, and tensor shape traces of original model.
                - "ablated_metrics": Corresponding metrics for modified architecture.
                - "mutated_code": Clean source script of ablated variation.
                - "diff_summary": Explanation of structural differences and computational impact.
                
        # TODO: Step 1: Deepcopy baseline_state to initialize ablated_state.
        # TODO: Step 2: Inject `mutation_instruction` directly into ablated_state['paper_context'] as an overriding architectural rule.
        # TODO: Step 3: Reset ablated_state['attempt_count'] = 0 and clear prior 'current_code' and 'execution_logs'.
        # TODO: Step 4: Execute self.orchestrator.run_translation_loop(ablated_state).
        # TODO: Step 5: Extract runtime logs and telemetry traces from both baseline and ablated runs.
        # TODO: Step 6: Construct and return comparative analysis payload formatted for frontend graphical charts.
        """
        print(f"--- [Ablation Controller] Executing architectural mutation: '{mutation_instruction}' ---")
        if baseline_state.get("execution_status") != "SUCCESS":
            raise ValueError("Ablation needs a baseline that passed the sandbox (execution_status == 'SUCCESS')")
        if not mutation_instruction.strip():
            raise ValueError("mutation_instruction is empty")

        baseline_code = baseline_state.get("current_code", "")

        # WHY critic_feedback/current_code and not paper_context: the graph starts at the Researcher,
        # which OVERWRITES paper_context. critic_feedback + current_code reach the Coder untouched
        # (Researcher/Architect never write them), so the mutation survives to where code is written.
        ablated_state = copy.deepcopy(baseline_state)
        ablated_state["attempt_count"] = 0
        ablated_state["execution_logs"] = []
        ablated_state["tensor_shape_traces"] = {}
        ablated_state["execution_status"] = "UNTESTED"
        ablated_state["current_code"] = baseline_code  # shown to the Coder as the code to modify
        ablated_state["critic_feedback"] = (
            "MANDATORY ABLATION: modify the previous working code as follows and change nothing else: "
            f"{mutation_instruction.strip()}"
        )

        ablated_final = await self.orchestrator.run_translation_loop(ablated_state)
        ablated_code = ablated_final.get("current_code", "")

        baseline_metrics = self._metrics(baseline_state)
        ablated_metrics = self._metrics(ablated_final)
        return {
            "baseline_metrics": baseline_metrics,
            "ablated_metrics": ablated_metrics,
            "mutated_code": ablated_code,
            "diff_summary": self._diff_summary(baseline_code, ablated_code, baseline_metrics, ablated_metrics),
        }

    @staticmethod
    def _metrics(state: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic metrics from what the sandbox actually measured (no LLM, no invented numbers)."""
        traces = state.get("tensor_shape_traces") or {}
        code = state.get("current_code", "") or ""
        return {
            "execution_status": state.get("execution_status", "UNKNOWN"),
            "attempts": state.get("attempt_count", 0),
            "traced_layers": len(traces),
            "layer_names": sorted(traces),
            "tensor_shape_traces": traces,
            "code_lines": len(code.splitlines()),
        }

    @staticmethod
    def _diff_summary(base_code: str, new_code: str, base_m: Dict[str, Any], new_m: Dict[str, Any]) -> str:
        diff = list(difflib.unified_diff(base_code.splitlines(), new_code.splitlines(), lineterm="", n=0))
        added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        gone = sorted(set(base_m["layer_names"]) - set(new_m["layer_names"]))
        new = sorted(set(new_m["layer_names"]) - set(base_m["layer_names"]))
        parts = [f"Code diff: +{added}/-{removed} lines.",
                 f"Traced layers: {base_m['traced_layers']} -> {new_m['traced_layers']}."]
        if gone:
            parts.append(f"Layers removed: {', '.join(gone)}.")
        if new:
            parts.append(f"Layers added: {', '.join(new)}.")
        if new_m["execution_status"] != "SUCCESS":
            parts.append(f"Ablated variant did NOT pass the sandbox ({new_m['execution_status']}).")
        return " ".join(parts)
