"""
Critic Agent (Static Analysis & Runtime Diagnostics Node)

Performs preliminary lightweight analysis (AST inspection, dependency whitelisting, flake8/mypy logic)
before expensive GPU sandbox execution. When execution fails, analyzes deep Docker tracebacks
to explain exact tensor dimension mismatches or CUDA out-of-memory bugs to the Coder Agent.
"""

import ast
import re
from typing import Dict, Any, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json
import re
from agents.state import SoulGraphState

from dotenv import load_dotenv
load_dotenv()


class CriticAgent:
    """
    LangGraph node acting as quality control, static code linter, and runtime error diagnostician.
    """

    def __init__(self, model_provider: str = "gemini"):
        """
        Initializes diagnostic LLM evaluation utilities.
        """
        self.provider = model_provider
        self.llm = ChatNVIDIA(
          model="nvidia/nemotron-3-ultra-550b-a55b",
          temperature=0.1,
          top_p=0.95,
          max_tokens=16384,
          chat_template_kwargs={"enable_thinking":True},
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a senior deep learning debugging assistant. Your job is to analyze PyTorch/JAX execution errors and provide exact, step-by-step reasoning on how to fix tensor shape mismatches or syntax bugs."),
            ("human", '''Broken Code:
{current_code}

Error Traceback from Sandbox:
{traceback}

Provide extremely concise instructions to the coder on exactly how to fix the code above to resolve the error. Do not write the code yourself.''')
        ])

    def run_static_analysis(self, code_string: str) -> Tuple[bool, str]:
        """
        Executes safe AST parsing and lint checks before sending code to the GPU sandbox.
        """
        whitelist = {"torch", "jax", "equinox", "einops", "numpy", "math", "typing"}
        try:
            tree = ast.parse(code_string)
            # ast.parse only checks GRAMMAR. Errors like duplicate argument names, `return` outside a
            # function, or `nonlocal` misuse surface at compile() (found live: an LLM-transpiled JAX
            # module with `def f(self, key, key)` passed ast.parse and was reported "syntax_valid").
            compile(code_string, "<generated>", "exec")
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except ValueError as e:                      # e.g. source contains null bytes
            return False, f"Syntax Error: {str(e)}"
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split('.')[0]
                    if base_module not in whitelist:
                        return False, f"Unauthorized import detected: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base_module = node.module.split('.')[0]
                    if base_module not in whitelist:
                        return False, f"Unauthorized import detected: {node.module}"
        return True, "Static analysis passed."

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph Node Execution Entrypoint for Critic Agent.
        """
        print("--- [Critic Agent] Evaluating code consistency & diagnosing tracebacks ---")
        current_code = state.get("current_code", "")
        
        # 1. Static Analysis (Catch Syntax Errors early)
        is_valid, msg = self.run_static_analysis(current_code)
        if not is_valid:
            return {
                "execution_status": "STATIC_ANALYSIS_FAILED",
                "critic_feedback": f"Static analysis failed before execution: {msg}. Please fix the syntax or remove unauthorized imports."
            }
            
        # 2. Dynamic execution analysis
        execution_logs = state.get("execution_logs", [])
        if not execution_logs:
            traceback_str = "No execution logs provided."
        else:
            traceback_str = execution_logs[-1].get("traceback", "No traceback provided.")
            
        chain = self.prompt | self.llm
        
        response = await chain.ainvoke({
            "current_code": current_code,
            "traceback": traceback_str
        })
        
        return {
            "critic_feedback": response.content
        }
