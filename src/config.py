from pathlib import Path

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
INITIAL_K = 10

OLLAMA_MODEL = "llama3.2"