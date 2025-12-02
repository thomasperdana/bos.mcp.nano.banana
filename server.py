from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import google.generativeai as genai
from langgraph_agent import FancylogoGeneratorAgent
import os
from typing import List
from PIL import Image
import io
import uuid
from langgraph.checkpoints import SqliteSaver

# Configure the Gemini API key
# Make sure to set the GOOGLE_API_KEY environment variable
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = FastAPI()

from upscaler_agent import UpscalerAgent

# Initialize the LangGraph agents
model = genai.GenerativeModel('gemini-pro-vision')
memory = SqliteSaver.from_conn_string(":memory:")
agent = FancylogoGeneratorAgent(model)
agent.graph = agent.build_graph(memory)
upscaler_agent = UpscalerAgent(model)


# In-memory storage for generated images (for simplicity)
# In a production environment, you might want to use a more robust storage solution
generated_images_store = {}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return JSONResponse(content={"file_path": file_path})

@app.post("/generate-prompt-suggestions")
async def generate_prompt_suggestions(prompt: str = Form(...)):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.graph.invoke({"prompt": prompt}, config)
    suggestions = result['messages'][-1].content.strip().split('\n')
    return JSONResponse(content={"suggestions": suggestions, "thread_id": thread_id})

@app.post("/generate-image")
async def generate_image(thread_id: str = Form(...), selected_prompt: str = Form(...), size: str = Form(...), count: int = Form(...), image_path: str = Form(None)):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.graph.invoke(None, config, **{"selected_prompt": selected_prompt, "image_size": size, "image_count": count, "image_path": image_path})
    image_id = str(uuid.uuid4())
    generated_images_store[image_id] = result['generated_images'][0]
    return JSONResponse(content={"image_id": image_id})

@app.get("/get-image/{image_id}")
async def get_image(image_id: str):
    # This endpoint will retrieve a generated image.
    image_data = generated_images_store.get(image_id)
    if image_data:
        return StreamingResponse(io.BytesIO(image_data), media_type="image/png")
    return JSONResponse(content={"error": "Image not found"}, status_code=404)

@app.post("/upscale-image")
async def upscale_image(image_id: str = Form(...)):
    image_data = generated_images_store.get(image_id)
    if not image_data:
        return JSONResponse(content={"error": "Image not found"}, status_code=404)

    result = upscaler_agent.run(image_data)
    upscaled_image_data = result["upscaled_image"]
    upscaled_image_id = f"upscaled_{image_id}"
    generated_images_store[upscaled_image_id] = upscaled_image_data
    return JSONResponse(content={"image_id": upscaled_image_id})

@app.get("/download-image/{image_id}")
async def download_image(image_id: str, format: str = "png"):
    # This endpoint will download an image in the specified format.
    image_data = generated_images_store.get(image_id)
    if not image_data:
        return JSONResponse(content={"error": "Image not found"}, status_code=404)

    image = Image.open(io.BytesIO(image_data))
    output = io.BytesIO()

    if format.lower() == "jpeg":
        image.save(output, format="JPEG")
        media_type = "image/jpeg"
    elif format.lower() == "pdf":
        image.save(output, format="PDF")
        media_type = "application/pdf"
    else:
        image.save(output, format="PNG")
        media_type = "image/png"

    output.seek(0)
    return StreamingResponse(output, media_type=media_type, headers={"Content-Disposition": f"attachment; filename=image.{format.lower()}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
