"""
Coder Agent (Tensor Operations Implementation Node)

Receives structural blueprints from the Architect and writes complete, functional,
and numerically stable backend code (PyTorch/JAX). If re-invoked in a feedback loop due to sandbox
runtime failures, it incorporates Critic error diagnostics to automatically resolve bugs.
"""
import os
from typing import Dict, Any, List

# TODO: Import LLM interfaces and code extraction utilities
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json
import re
from agents.state import SoulGraphState

from dotenv import load_dotenv
load_dotenv()
class CoderAgent:
    """
    LangGraph node responsible for precise tensor calculus coding and automatic bug remediation.
    """

    def __init__(self, model_provider: str = "gemini"):
        """
        Initializes specialized code generation LLM settings (low temperature for deterministic syntax).
        # TODO: Initialize Coder LLM chain with code-specific system prompt.
        """
        self.provider = model_provider
        # TODO: Configure code implementation prompt template
        self.llm =ChatNVIDIA(
          model="nvidia/nemotron-3-ultra-550b-a55b",
          temperature=0.1,
          top_p=0.95,
          max_tokens=16384,
          chat_template_kwargs={"enable_thinking":True},
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert PyTorch/JAX engineer. Your job is to take an architectural skeleton and implement the dense tensor mathematics inside the forward passes. Return ONLY the complete, runnable python code wrapped in a ```python block. HARNESS CONTRACT (your code is run automatically, nothing else is supplied): (1) the main nn.Module class is instantiated with NO arguments, so give EVERY __init__ parameter a default taken from the paper's hyperparameters (e.g. d_model=512, h=8); (2) define the top-level model class LAST, after any helper modules; (3) forward() receives the dummy inputs from the skeleton, by keyword name; inputs whose name contains 'mask' are boolean tensors (True = position is masked out), inputs named like ids/tokens/indices are int64."),
            ("human", """
            Target Component: {target_component}
            Architectural Skeleton:
            {structural_skeleton}
            Mathematical Formulas to Implement:
            {relevant_formulas}
            --- DEBUGGING CONTEXT (If Applicable) ---
            Previous Failed Code:
            {previous_code}
            Critic Feedback:
            {critic_feedback}
            Implement the full logic. Do not truncate.
            """)
        ])

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph Node Execution Entrypoint for Coder Agent.
        
        Args:
            state (SoulGraphState): State containing skeleton, attempt counts, and potentially error logs.
            
        Returns:
            Dict[str, Any]: Partial state update containing `current_code` and incrementing `attempt_count`.
            
        # TODO: Check state['attempt_count']. If > 0, we are in a debugging correction loop!
        # TODO: If debugging, pull `critic_feedback` and bottom frames of `execution_logs` to feed into prompt.
        # TODO: If initial run, feed `structural_skeleton` and `relevant_formulas` to generate complete forward() logic.
        # TODO: Require LLM to preserve all `# Implements: <Node_ID: Eq_3>` comments for Explainer UI linking.
        # TODO: Apply "Temperature Annealing" (if attempt_count > 2, increase LLM temperature slightly to force diverse solutions).
        # TODO: Extract clean Python script from Markdown code blocks in LLM response.
        # TODO: Return update dict with incremented attempt_count and updated current_code.
        """

        current_attempt = state.get("attempt_count", 0) + 1
        print(f"--- [Coder Agent] Generating backend tensor operations (Attempt {current_attempt}/5) ---")

        structural_skeleton = state.get("structural_skeleton","")
        relevant_formulas = state.get("relevant_formulas",[])
        target_component = state.get("target_component","")

        previous_code = state.get("current_code","")
        critic_feedback = state.get("critic_feedback","")
        chain = self.prompt | self.llm

        response = await chain.ainvoke({
            'structural_skeleton': structural_skeleton,
            'relevant_formulas': relevant_formulas,
            'target_component': target_component,
            'previous_code': previous_code,
            'critic_feedback': critic_feedback,

        })
        raw_text = response.content

        code_match = re.search(r'```python(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
        parsed_code = code_match.group(1).strip() if code_match else "# Parsing Error: No code found"

        json_match = re.search(r'```json(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
        try:
            parsed_json = json.loads(json_match.group(1).strip()) if json_match else {}
        except json.JSONDecodeError:
            parsed_json = {}




        return {
            "current_code": parsed_code,
            "attempt_count": current_attempt,  # (This is already calculated for you at the top of the function)
            "execution_status": "UNTESTED"
        }
