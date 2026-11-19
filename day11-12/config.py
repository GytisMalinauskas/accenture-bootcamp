import os

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b-q4_K_M")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bootcamp_docs"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150
TOP_K = 8

SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided context.
Use ONLY the context below to answer. If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context}
"""

TEMPERATURE = 0.3
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
MAX_TOKENS = 1024
TOKEN_LIMIT_PARAM = os.getenv("TOKEN_LIMIT_PARAM", "max_tokens")
