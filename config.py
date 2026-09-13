import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directories
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configurations
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_REASONING_EFFORT = "minimal"
LLM_MAX_TOKENS = 2000

# Text Splitter settings
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 600

# Retriever settings
RETRIEVER_K = 5
