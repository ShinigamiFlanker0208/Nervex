"""
Tool-Calling Explainer Chat Agent (Agentic RAG Assistant)

Interactive frontend conversational assistant equipped with specialized tool calling capabilities.
Allows users to highlight generated backend code and query theoretical foundations
(e.g., "Why is this GELU activation here?"). The agent dynamically calls tools to query AST mappings,
Neo4j Graph neighborhoods, and Vector DB passages before formatting a mathematically rigorous explanation.
"""

from typing import List, Dict, Any, Optional, Tuple

# TODO: Import LangChain tool calling agents and schemas
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.tools import tool, StructuredTool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq

# TODO: Import project database engines and AST parser tools
# from explainer.ast_parser import CodeTheoryMapper
# from db.graph_db import Neo4jGraphManager
# from db.vector_db import VectorStoreManager


# =====================================================================
# Explainer Agent Tool Definitions
# =====================================================================

# TODO: Define LangChain @tool decorators for agent tool-calling capabilities:

# @tool
def query_knowledge_graph(node_id: str, depth: int = 2) -> Dict[str, Any]:
    """
    Fetches theoretical entity properties and surrounding structural relationships from Neo4j Knowledge Graph.
    Use this tool when you have an entity ID (e.g., 'GELU_Activation_01') and need to know which formula defines it or what component feeds into it.
    # TODO: Connect to Neo4jGraphManager and execute query_graph_neighborhood(node_id, depth).
    """
    from api.deps import get_graph_db
    depth = max(1, min(int(depth), 4))  # depth is interpolated into Cypher by the manager: clamp it
    return get_graph_db().query_graph_neighborhood(node_id, depth)


# @tool
def search_paper_text(query_string: str, n_results: int = 3) -> List[str]:
    """
    Searches original research paper markdown chunks in Vector DB for relevant textual descriptions and justifications.
    Use this tool when you need explanatory intuition or author rationale (e.g., "Why did authors choose GELU over ReLU?").
    # TODO: Connect to VectorStoreManager and execute similarity_search(query_string, n_results).
    """
    from api.deps import get_vector_db
    hits = get_vector_db().similarity_search(query_string, n_results=max(1, min(int(n_results), 8)))
    return [h.get("content") or h.get("document") or str(h) for h in hits]


# @tool
def get_ast_lineage_for_lines(start_line: int, end_line: int, code_script: str) -> Dict[str, Any]:
    """
    Analyzes Abstract Syntax Tree of generated script between specified line numbers to retrieve linked Graph IDs and tensor shapes.
    Use this tool FIRST whenever a user asks a question about a highlighted block of source code.
    # TODO: Instantiate CodeTheoryMapper on code_script, execute parse_and_map(), and call get_context_for_line_range().
    """
    from explainer.ast_parser import CodeTheoryMapper
    mapper = CodeTheoryMapper(code_script)
    mapper.parse_and_map()
    return mapper.get_context_for_line_range(start_line, end_line)


# Captured at import: the schemas the LLM sees must not change if someone monkeypatches the module names.
_TOOL_SPECS = {
    "query_knowledge_graph": query_knowledge_graph,
    "search_paper_text": search_paper_text,
}


# =====================================================================
# Explainer Agent Orchestrator
# =====================================================================

class ExplainerChatAgent:
    """
    Orchestrates conversational tool-calling agent loop for interactive frontend explanations.
    """

    MAX_TOOL_HOPS = 4

    SYSTEM_PROMPT = (
        "You are a senior ML engineer explaining generated model code to its owner. "
        "Ground every claim in the paper: use the tools to look up the knowledge-graph node and the "
        "paper passages before answering. Never invent equation numbers. If the tools return nothing "
        "relevant, say so plainly. Reply in Markdown, with LaTeX for math and explicit tensor shapes."
    )

    def __init__(self, model_provider: str = "nvidia", llm: Any = None):
        """
        `llm` is injectable so the loop is unit-testable without network. By default builds the
        same ChatNVIDIA model family the rest of Nervex uses (no model change).
        """
        self.provider = model_provider
        if llm is None:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            llm = ChatNVIDIA(model="nvidia/nemotron-3-ultra-550b-a55b", temperature=0.2, max_tokens=4096)
        self.llm = llm
        self._tools = {
            "query_knowledge_graph": query_knowledge_graph,
            "search_paper_text": search_paper_text,
        }

    def _bound_llm(self):
        from langchain_core.tools import StructuredTool
        tools = [StructuredTool.from_function(fn) for fn in _TOOL_SPECS.values()]
        return self.llm.bind_tools(tools)

    async def explain(self, user_question: str, highlighted_code_lines: Tuple[int, int], current_script: str,
                      chat_history: List[Any]) -> Dict[str, Any]:
        """Full result: answer + citations. `answer_user_query` is the string-only view of this."""
        import asyncio
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

        start, end = highlighted_code_lines
        # Deterministic first hop (no LLM): which graph nodes / shapes does this selection touch?
        try:
            lineage = get_ast_lineage_for_lines(start, end, current_script)
        except SyntaxError as exc:
            lineage = {"node_ids": [], "tensor_shapes": [], "segments": [],
                       "selected_code": "", "note": f"script does not parse: {exc}"}

        messages: List[Any] = [SystemMessage(content=self.SYSTEM_PROMPT)]
        for turn in chat_history[-10:]:
            role, content = turn.get("role"), turn.get("content", "")
            messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
        messages.append(HumanMessage(content=(
            f"Selected lines {start}-{end}:\n```python\n{lineage.get('selected_code', '')}\n```\n"
            f"Static lineage: graph nodes={lineage['node_ids']} shapes={lineage['tensor_shapes']}\n\n"
            f"Question: {user_question}"
        )))

        llm = self._bound_llm()
        cited_nodes: List[str] = list(lineage["node_ids"])
        passages: List[str] = []
        response = None
        for _ in range(self.MAX_TOOL_HOPS + 1):
            response = await llm.ainvoke(messages)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                break
            messages.append(response)
            for call in calls:
                fn = self._tools.get(call["name"])
                if fn is None:
                    out: Any = f"unknown tool {call['name']}"
                else:
                    try:  # tools hit Neo4j/Chroma (blocking): keep the event loop free
                        out = await asyncio.to_thread(fn, **call["args"])
                    except Exception as exc:  # tell the model, don't crash the chat
                        out = f"tool error: {type(exc).__name__}: {exc}"
                if call["name"] == "search_paper_text" and isinstance(out, list):
                    passages.extend(str(x) for x in out)
                if call["name"] == "query_knowledge_graph" and isinstance(out, dict) and call["args"].get("node_id"):
                    cited_nodes.append(call["args"]["node_id"])
                messages.append(ToolMessage(content=str(out)[:6000], tool_call_id=call["id"]))
        else:
            # hop budget exhausted while the model still wanted tools
            pass

        answer = (response.content if response is not None else "") or "No answer produced."
        if getattr(response, "tool_calls", None):
            answer += "\n\n_(Stopped: tool-call budget exhausted.)_"
        return {
            "answer_markdown": answer if isinstance(answer, str) else str(answer),
            "cited_graph_nodes": list(dict.fromkeys(cited_nodes)),
            "cited_paper_passages": passages,
            "inferred_tensor_shapes": {f"shape_{i}": sh for i, sh in enumerate(lineage["tensor_shapes"])} or None,
        }

    async def answer_user_query(self, user_question: str, highlighted_code_lines: Tuple[int, int], current_script: str, chat_history: List[Any]) -> str:
        """String-only view of `explain()`."""
        print(f"--- [Explainer Agent] Processing inquiry for lines {highlighted_code_lines}: '{user_question}' ---")
        result = await self.explain(user_question, highlighted_code_lines, current_script, chat_history)
        return result["answer_markdown"]
