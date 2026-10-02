"""
Shared retrieval logic used by both sub-agents.

Uses a local HuggingFace embedding model (runs on CPU, no API key,
no cost) instead of a paid embeddings API.
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "./data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Instantiate once at module scope so the model weights and Chroma client
# are loaded a single time per process, not on every retrieve() call.
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
_chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)


def retrieve(query: str, collection_name: str, k: int = 4) -> list[str]:
    """
    Similarity search against the given Chroma collection.
    Returns a list of plain-text chunks.
    """
    vectordb = Chroma(
        client=_chroma_client,
        collection_name=collection_name,
        embedding_function=_embeddings,
    )
    docs = vectordb.similarity_search(query, k=k)
    return [d.page_content for d in docs]