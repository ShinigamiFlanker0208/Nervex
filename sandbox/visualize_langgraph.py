import sys
from pathlib import Path

# Add project root to path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, Literal

def visualize_workflow():
    print("Building LangGraph Architecture Visualization...")
    
    # We create a dummy graph with the exact same structure to avoid 
    # NotImplementedErrors from your unbuilt agent classes!
    builder = StateGraph(dict)
    
    # Dummy nodes
    builder.add_node("researcher", lambda x: x)
    builder.add_node("architect", lambda x: x)
    builder.add_node("coder", lambda x: x)
    builder.add_node("sandbox_executor", lambda x: x)
    builder.add_node("critic", lambda x: x)
    
    # Edges
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "architect")
    builder.add_edge("architect", "coder")
    builder.add_edge("coder", "sandbox_executor")
    
    # Conditional edge representation
    builder.add_conditional_edges(
        "sandbox_executor", 
        lambda x: "critic", # Dummy condition
        {"critic": "critic", END: END}
    )
    builder.add_edge("critic", "coder")
    
    workflow = builder.compile()
    
    print("Generating Graph Visualization...")
    try:
        png_bytes = workflow.get_graph().draw_mermaid_png()
        output_path = Path(__file__).parent / "langgraph_architecture.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ Workflow Architecture Visualization saved successfully to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to generate visualization: {e}")

if __name__ == "__main__":
    visualize_workflow()
