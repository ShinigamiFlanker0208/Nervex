import os
import sys
import asyncio
import gradio as gr

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import SoulGraphOrchestrator

# Initialize the LangGraph Orchestrator
print("Initializing Orchestrator...")
orchestrator = SoulGraphOrchestrator()
workflow = orchestrator.build_graph()

async def generate_architecture(component: str, framework: str):
    """
    Async generator that feeds LangGraph state updates to the Gradio UI in real-time.
    """
    initial_state = {
        "target_component": component,
        "target_framework": framework,
        "paper_context": "",
        "relevant_formulas": [],
        "hyperparameters": {},
        "structural_skeleton": "",
        "expected_inputs": {},
        "current_code": "",
        "attempt_count": 0,
        "execution_status": "PENDING",
        "critic_feedback": "",
        "execution_logs": []
    }
    
    log_output = f"### 🚀 Starting Swarm Execution\n**Target Component:** {component}\n**Framework:** {framework}\n\n---\n"
    final_code = ""
    
    yield log_output, final_code

    try:
        # We use .astream to get real-time node updates
        async for event in workflow.astream(initial_state):
            for node_name, state_update in event.items():
                log_output += f"\n✅ **Node Finished:** `{node_name}`\n"
                
                # If Critic gave feedback, show it
                if node_name == "critic" and "critic_feedback" in state_update:
                    log_output += f"> **Critic Says:** {state_update['critic_feedback']}\n"
                
                # If Sandbox failed, show attempts
                if node_name == "sandbox_executor":
                    status = state_update.get("execution_status", "")
                    log_output += f"> **Sandbox Status:** `{status}`\n"
                    
                # If Coder wrote code, update the live code block
                if "current_code" in state_update and state_update["current_code"]:
                    final_code = f"```python\n{state_update['current_code']}\n```"
                
                yield log_output, final_code
                
        log_output += "\n\n🎉 **Swarm Execution Complete!**"
        yield log_output, final_code
        
    except Exception as e:
        log_output += f"\n\n❌ **CRITICAL ERROR:** {str(e)}"
        yield log_output, final_code


# ---------------------------------------------------------------------------
# Gradio Layout Definition
# ---------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🧠 Nervex AI Architect Swarm")
    gr.Markdown("Input a deep learning component and watch the multi-agent swarm research, architect, code, and debug it in real-time.")
    
    with gr.Row():
        with gr.Column(scale=1):
            component_input = gr.Textbox(label="Target Component", placeholder="e.g. SwiGLU Activation, Multi-Head Attention", value="SwiGLU Activation")
            framework_input = gr.Dropdown(choices=["PyTorch", "JAX/Equinox"], label="Target Framework", value="PyTorch")
            generate_btn = gr.Button("🚀 Generate Architecture", variant="primary")
            
        with gr.Column(scale=2):
            logs_output = gr.Markdown(label="Swarm Live Logs")
            
    with gr.Row():
        code_output = gr.Markdown(label="Final Generated Code")

    # Connect the button to the generator
    generate_btn.click(
        fn=generate_architecture,
        inputs=[component_input, framework_input],
        outputs=[logs_output, code_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
