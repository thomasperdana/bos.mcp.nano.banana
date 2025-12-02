import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    try:
        # Configure GenAI SDK
        genai.configure(api_key=api_key)
        print("Google Generative AI configured successfully.")

        # Initialize LangChain Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
        )
        print("ChatGoogleGenerativeAI initialized successfully.")
        print("Dependency test passed!")

    except Exception as e:
        print(f"Dependency test failed with an error: {e}")

try:
    from google.ai.generativelanguage import FileData
    import langchain_google_genai
    print("✅ Dependencies are compatible. FileData object found.")
except ImportError as e:
    print(f"❌ Error: {e}")
