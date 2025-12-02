from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

class FancylogoGeneratorAgent:
    def __init__(self, model):
        self.model = model
        self.graph = self.build_graph()

    def build_graph(self, memory=None):
        class AgentState(TypedDict):
            prompt: str
            optimized_prompts: List[str]
            selected_prompt: str
            image_size: str
            image_count: int
            generated_images: List[bytes]
            image_format: str
            image_path: str

        graph = StateGraph(AgentState)
        graph.add_node("generate_prompt_suggestions", self.generate_prompt_suggestions)
        graph.add_node("generate_image", self.generate_image)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self.router,
            {
                "continue": "generate_image",
                "generate": "generate_prompt_suggestions",
            },
        )
        graph.add_edge("generate_prompt_suggestions", END)
        graph.add_edge("generate_image", END)

        return graph.compile(checkpointer=memory, interrupt_after=["generate_prompt_suggestions"])

    def router(self, state):
        if "selected_prompt" in state and state["selected_prompt"]:
            return "continue"
        else:
            return "generate"

    def generate_image(self, state):
        prompt = state["selected_prompt"]
        image_size = state.get("image_size")
        image_count = state.get("image_count", 1)

        prompt_parts = [f"{prompt}. Generate an image with the following size: {image_size}"]

        if "image_path" in state and state["image_path"]:
            with open(state["image_path"], "rb") as image_file:
                image_data = image_file.read()
            prompt_parts.append({"inline_data": {"data": image_data, "mime_type": "image/png"}})

        generated_images = []
        for _ in range(image_count):
            response = self.model.generate_content(prompt_parts)
            image_data = response.parts[0].inline_data.data
            generated_images.append(image_data)

        return {"generated_images": generated_images}


    def generate_prompt_suggestions(self, state):
        prompt = state["prompt"]
        response = self.model.generate_content(
            f"Generate 4 improved and optimized prompts for image generation based on the following prompt: '{prompt}'"
        )
        suggestions = response.text.strip().split('\n')
        return {"optimized_prompts": suggestions}

    def run(self, prompt: str):
        return self.graph.invoke({"prompt": prompt})

