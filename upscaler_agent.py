from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import operator

class UpscalerAgent:
    def __init__(self, model):
        self.model = model
        self.graph = self.build_graph()

    def build_graph(self, memory=None):
        class AgentState(TypedDict):
            image: bytes
            upscaled_image: bytes

        graph = StateGraph(AgentState)
        graph.add_node("upscale_image", self.upscale_image)
        graph.set_entry_point("upscale_image")
        graph.add_edge("upscale_image", END)
        return graph.compile(checkpointer=memory)

    def upscale_image(self, state):
        image_data = state["image"]
        prompt = [
            "Upscale the following image to a higher resolution:",
            {"inline_data": {"data": image_data, "mime_type": "image/png"}},
        ]
        response = self.model.generate_content(prompt)
        upscaled_image_data = response.parts[0].inline_data.data
        return {"upscaled_image": upscaled_image_data}

    def run(self, image: bytes):
        return self.graph.invoke({"image": image})
