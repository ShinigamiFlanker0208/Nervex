"""
Architect Agent (Model Blueprint & Tensor Shape Designer Node)

Translates retrieved math equations and graph lineages into an object-oriented module skeleton,
enforcing explicit tensor dimensionality annotations (e.g., `# [B, H, S, D] -> [B, S, H*D]`) above
every planned layer to eliminate tensor hallucination errors downstream.
"""

import os
from typing import Dict, Any

# TODO: Import LLM interfaces (Gemini / Groq / OpenAI)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re
from agents.state import SoulGraphState
from langchain_nvidia_ai_endpoints import ChatNVIDIA



class ArchitectAgent:
    """
    LangGraph node responsible for generating high-level structural blueprints and tensor shape contracts.
    """
    def __init__(self, model_provider: str = "gemini"):
        """
        Initializes high-reasoning LLM instances (preferably Gemini 1.5 Pro for math/shape reasoning).
        # TODO: Initialize prompt templates and LLM execution chain.
        """

        self.llm =ChatNVIDIA(
          model="nvidia/nemotron-3-ultra-550b-a55b",
          api_key=os.environ.get("NVIDIA_API_KEY"),  # was the literal string "$NVIDIA_API_KEY" (never expanded -> 401)
          temperature=0.1,
          top_p=0.95,
          max_tokens=16384,
          chat_template_kwargs={"enable_thinking":True},
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an AI deep learning architect. You do NOT write math implementation. You ONLY write structural class skeletons with exact tensor dimensionality tracking."),
            ("human", """
                   Build a {target_framework} skeleton for: {target_component}.

                   Here is the math: {relevant_formulas}
                   Here are the default hyperparameters: {hyperparameters}
                   Here is the contextual graph text: {paper_context}

                   Return your response strictly in two blocks:
                   1. A Python code block with the class skeleton.
                   2. A JSON block mapping the expected input tensor shapes (e.g. {{"x": [32, 12, 512]}}). Use the EXACT forward() argument names as keys, small shapes (batch <= 4, sequence <= 128), and append a dtype word only for non-float inputs (e.g. {{"mask": "[4, 1, 1, 128] bool", "ids": "[4, 128] int64"}}). In the skeleton, give every __init__ parameter a default from the paper hyperparameters so the class can be built with no arguments.
                   """)
        ])

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph Node Execution Entrypoint for Architect Agent.
        
        Args:
            state (SoulGraphState): Current state containing research context and formula lists.
            
        Returns:
            Dict[str, Any]: Partial state update with `structural_skeleton` and `expected_inputs`.
            
        # TODO: Formulate prompt combining paper_context, relevant_formulas, and hyperparameters.
        # TODO: Instruct LLM to generate a clean PyTorch `nn.Module` (or JAX Equinox) Class Skeleton without internal mathematical implementation.
        # TODO: REQUIRE LLM to write precise tensor dimensions as markdown comments above every layer declaration.
        # TODO: REQUIRE LLM to output a dict of synthetic dummy input tensor shapes (e.g., `{"x": "(32, 128, 512)"}`) for sandbox validation.
        # TODO: Parse output and return partial state update.
        """
        print("--- [Architect Agent] Drafting structural skeleton with explicit tensor shape contracts ---")

        paper_context=state.get("paper_context")
        relevant_formulas=state.get("relevant_formulas")
        hyperparameters=state.get("hyperparameters")
        target_component=state.get("target_component")
        target_framework=state.get("target_framework")

        chain=self.prompt|self.llm

        response= await chain.ainvoke({
            "target_component": target_component,
            "paper_context": paper_context,
            "relevant_formulas": relevant_formulas,
            "hyperparameters": hyperparameters,
            "target_framework": target_framework
        }
        )
        raw_text = response.content

        code_match = re.search(r'```python(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
        parsed_code = code_match.group(1).strip() if code_match else "# Parsing Error: No code found"

        json_match = re.search(r'```json(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
        try:
            parsed_json = json.loads(json_match.group(1).strip()) if json_match else {}
        except json.JSONDecodeError:
            parsed_json = {}

        return {
                "structural_skeleton": parsed_code,
                "expected_inputs": parsed_json
            }

