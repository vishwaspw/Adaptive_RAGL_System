import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vector store configuration
VECTOR_DB_PATH = "data/vector_db"

# Model settings
MODEL_NAME = "models/gemini-1.5-pro"
MAX_TOKENS = 4096
TEMPERATURE = 0.2

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Learning settings
AUTO_UPDATE_THRESHOLD = 0.75  # Confidence threshold for auto-updating the vector DB
LEARNING_RATE = 0.1  # Weight for new knowledge updates 