import os
import base64
import time
from pathlib import Path
from dotenv import load_dotenv

# MCP Imports
from mcp.server.fastmcp import FastMCP

# LangChain Imports (For Text & Logic)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Google GenAI Imports (For Image Generation)
import google.generativeai as genai
from PIL import Image
import io

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure GenAI SDK
genai.configure(api_key=api_key)

# 2. Initialize MCP Server
mcp = FastMCP("Gemini Creative Agent")

# 3. Initialize LangChain Model (The Logic Brain)
# We use this for chat and for optimizing prompts
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", # Powerful reasoning model
    temperature=0.7,
)

# --- HELPER: Image Storage ---
# We need a place to save generated images so you can see them.
IMAGE_DIR = Path.cwd() / "generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

# --- TOOL 1: General Chat ---
@mcp.tool()
def ask_gemini(prompt: str) -> str:
    """
    Ask Gemini a general question or for advice.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- TOOL 2: Prompt Optimizer ---
@mcp.tool()
def optimize_image_prompt(rough_idea: str) -> str:
    """
    Takes a rough image idea and rewrites it into a highly detailed, 
    optimized prompt suitable for an AI image generator.
    """
    system_instruction = (
        "You are an expert Prompt Engineer for AI Image Generators (like Imagen 3, Midjourney). "
        "Rewrite the user's rough idea into a detailed, descriptive prompt. "
        "Include details about lighting, style (photorealistic, oil painting, etc), "
        "composition, camera angle, and color palette. "
        "Output ONLY the optimized prompt text."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{rough_idea}")
    ])
    
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        optimized = chain.invoke({"rough_idea": rough_idea})
        return f"OPTIMIZED PROMPT:\n{optimized}"
    except Exception as e:
        return f"Error optimizing prompt: {str(e)}"

# --- TOOL 3: Image Generator ---
@mcp.tool()
def generate_image(prompt: str, filename_prefix: str = "img") -> str:
    """
    Generates an image using Google's Imagen model based on the prompt.
    Saves the image locally and returns the file path.
    
    Args:
        prompt: The detailed description of the image to generate.
        filename_prefix: A short name for the file (default: "img")
    """
    try:
        # Use the Imagen model directly via GenAI SDK
        # Note: You can change this to 'models/gemini-3-pro-image-preview' if you have access
        imagen_model = genai.GenerativeModel("imagen-3.0-generate-001")
        
        result = imagen_model.generate_images(
            prompt=prompt,
            number_of_images=1,
        )
        
        if not result.images:
            return "No images were returned by the API."

        # Save the image
        timestamp = int(time.time())
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = IMAGE_DIR / filename
        
        # The result.images[0] is usually a PIL Image object or bytes depending on version
        # We ensure it's saved correctly
        image_data = result.images[0]
        image_data.save(filepath)
        
        return f"Image generated successfully! Saved at:\n{filepath}"

    except Exception as e:
        return f"Error generating image: {str(e)}"

if __name__ == "__main__":
    mcp.run()