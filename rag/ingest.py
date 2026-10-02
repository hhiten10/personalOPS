"""
Ingestion module for personalOPS.

Primary path:  add_entry() — called by the agent's write node
Utility path:  ingest_documents() — batch file import (kept as a fallback)
One-time use:  clear_collection() — wipe a collection clean

Usage (one-time wipe):
    python -m rag.ingest --clear
"""

import os
import re
from datetime import date as date_type

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "./data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
_chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _today() -> str:
    return date_type.today().isoformat()


def _extract_date_from_path(filepath: str) -> str:
    match = _DATE_RE.search(os.path.basename(filepath))
    return match.group() if match else _today()


def _get_vectordb(collection_name: str) -> Chroma:
    return Chroma(
        client=_chroma_client,
        collection_name=collection_name,
        embedding_function=_embeddings,
    )


# ── Wipe ─────────────────────────────────────────────────────────────

def clear_collection(collection_name: str):
    """Delete all documents from a collection."""
    _chroma_client.delete_collection(collection_name)
    # Recreate empty so future add_entry calls don't fail
    _chroma_client.get_or_create_collection(collection_name)
    print(f"Cleared collection '{collection_name}'.")


# ── Primary path: single-entry insert (chat-driven) ─────────────────

def add_entry(
    collection_name: str,
    doc_type: str,
    text: str,
    date: str | None = None,
    source: str = "chat_entry",
):
    """
    Embed and store a single piece of text with metadata.

    Args:
        collection_name: "gym_logs" or "study_notes"
        doc_type:        "workout_log", "lecture_notes", etc.
        text:            the raw content to embed
        date:            YYYY-MM-DD string; defaults to today
        source:          label for where this came from
    """
    chunks = _splitter.split_text(text)

    metadata = {
        "source_file": source,
        "date": date or _today(),
        "doc_type": doc_type,
    }

    vectordb = _get_vectordb(collection_name)
    vectordb.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))

    print(f"Added {len(chunks)} chunk(s) to '{collection_name}' "
          f"(doc_type='{doc_type}', date='{metadata['date']}').")


# ── Utility path: batch file import (fallback) ──────────────────────

def ingest_documents(
    source_dir: str,
    collection_name: str,
    doc_type: str,
    glob: str = "**/*.txt",
):
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"'{source_dir}' doesn't exist yet — create it and drop your "
            f"files in before running ingestion."
        )

    loader = DirectoryLoader(source_dir, glob=glob, loader_cls=TextLoader)
    raw_docs = loader.load()
    chunks = _splitter.split_documents(raw_docs)

    for chunk in chunks:
        filepath = chunk.metadata.get("source", "")
        chunk.metadata["source_file"] = os.path.basename(filepath)
        chunk.metadata["date"] = _extract_date_from_path(filepath)
        chunk.metadata["doc_type"] = doc_type

    vectordb = _get_vectordb(collection_name)
    vectordb.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks from '{source_dir}' "
          f"into '{collection_name}' (doc_type='{doc_type}').")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--clear" in sys.argv:
        clear_collection("gym_logs")
        clear_collection("study_notes")
        print("Both collections wiped. Add new data through chat.")
    else:
        ingest_documents("data/workout_logs", "gym_logs", doc_type="workout_log")
        ingest_documents("data/study_notes", "study_notes", doc_type="lecture_notes")