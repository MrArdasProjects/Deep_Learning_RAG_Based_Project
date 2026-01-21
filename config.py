import os

# Gemini API Configuration
GOOGLE_API_KEY = ""

# Set environment variable
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.1
TOP_K = 10  # Number of relevant chunks to retrieve


