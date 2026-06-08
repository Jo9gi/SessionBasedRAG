import os
from langchain_community.vectorstores import FAISS
from .embedding import embedding_model
from langchain_core.documents import Document
import pickle

FAISS_DIR = "faiss_indices"


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _index_path(session_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{session_id}.index")


def create_vector_store(chunks, session_id: str):
    """Create or update a FAISS index for a session.
    Args:
        chunks (list[Document]): List of LangChain Document objects.
        session_id (str): Identifier for the session (used for persistence).
    Returns:
        FAISS: FAISS vector store instance.
    """
    _ensure_dir(FAISS_DIR)
    # Build FAISS index from documents
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Persist index and metadata
    index_path = _index_path(session_id)
    vector_store.save_local(index_path)
    return vector_store


def load_vector_store(session_id: str):
    """Load an existing FAISS index for a session, or create an empty one if missing."""
    index_path = _index_path(session_id)
    if os.path.isdir(index_path):
        return FAISS.load_local(index_path, embedding_model)
    else:
        # Return an empty FAISS store (no documents yet)
        return FAISS.from_documents([], embedding_model)
