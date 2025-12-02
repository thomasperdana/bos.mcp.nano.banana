from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# --- State Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        rough_idea: The initial user prompt.
        optimized_prompt: The prompt after being optimized by the LLM.
        image_path: The path to the generated image.
        error: A potential error message.
    """
    rough_idea: str
    optimized_prompt: str
    image_path: str
    error: str

# --- Node Definitions ---

def optimize_prompt_node(state: GraphState, optimizer_tool) -> GraphState:
    """
    Optimizes the initial user idea into a detailed prompt.

    Args:
        state: The current graph state.
        optimizer_tool: The function to call for prompt optimization.

    Returns:
        A dictionary with the updated 'optimized_prompt' or 'error'.
    """
    try:
        rough_idea = state["rough_idea"]
        optimized_prompt = optimizer_tool(rough_idea)
        # The tool returns a string like "OPTIMIZED PROMPT:\n...", we want to extract the part after the newline
        if "OPTIMIZED PROMPT:" in optimized_prompt:
            optimized_prompt = optimized_prompt.split("\n", 1)[1]
        return {"optimized_prompt": optimized_prompt, "error": None}
    except Exception as e:
        return {"error": f"Error in optimize_prompt_node: {e}"}


def generate_image_node(state: GraphState, generator_tool) -> GraphState:
    """
    Generates an image based on the optimized prompt.

    Args:
        state: The current graph state.
        generator_tool: The function to call for image generation.

    Returns:
        A dictionary with the 'image_path' or 'error'.
    """
    try:
        optimized_prompt = state["optimized_prompt"]
        # The tool expects a filename_prefix, let's create one from the idea
        filename_prefix = state["rough_idea"][:20].replace(" ", "_")
        image_path = generator_tool(prompt=optimized_prompt, filename_prefix=filename_prefix)
        return {"image_path": image_path, "error": None}
    except Exception as e:
        return {"error": f"Error in generate_image_node: {e}"}

# --- Graph Definition ---

def create_graph(optimizer_tool, generator_tool):
    """
    Creates and compiles the LangGraph workflow.
    """
    workflow = StateGraph(GraphState)

    # We need to use functools.partial to pass the tools to the nodes
    from functools import partial

    # Add nodes to the workflow
    workflow.add_node("optimizer", partial(optimize_prompt_node, optimizer_tool=optimizer_tool))
    workflow.add_node("generator", partial(generate_image_node, generator_tool=generator_tool))

    # Set the entrypoint
    workflow.set_entry_point("optimizer")

    # Add edges
    workflow.add_edge("optimizer", "generator")
    workflow.add_edge("generator", END)

    # Compile the workflow
    graph = workflow.compile()
    return graph
