import os
import pickle

from langchain_community.vectorstores import FAISS
from .embedding import embedding_model

FAISS_DIR = "faiss_indices"


# =====================================================
# DIRECTORY HELPERS
# =====================================================

def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _index_path(session_id: str) -> str:
    """
    Session storage directory.

    Example:
    faiss_indices/
        session_id.index/
            index.faiss
            index.pkl
            chunks.pkl
            bm25.pkl
    """
    return os.path.join(
        FAISS_DIR,
        f"{session_id}.index"
    )


def _chunks_path(session_id: str) -> str:
    return os.path.join(_index_path(session_id),"chunks.pkl")


def _bm25_path(session_id: str) -> str:
    return os.path.join(_index_path(session_id),
        "bm25.pkl"
    )


# =====================================================
# FAISS VECTOR STORE
# =====================================================

def create_vector_store(chunks, session_id: str):
    """
    Create and persist FAISS vector store.
    """

    _ensure_dir(FAISS_DIR)

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    index_path = _index_path(session_id)

    vector_store.save_local(index_path)

    return vector_store


def load_vector_store(session_id: str):
    """
    Load FAISS vector store.
    """

    index_path = _index_path(session_id)

    if os.path.isdir(index_path):

        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    return None


# =====================================================
# CHUNKS PERSISTENCE
# =====================================================

def save_chunks(session_id, chunks):
    """
    Save chunked documents to disk.
    """
    index_path = _index_path(session_id)
    _ensure_dir(index_path)
    
    with open(_chunks_path(session_id), "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(session_id):
    """
    Load saved chunks.
    """
    path = _chunks_path(session_id)
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================
# BM25 PERSISTENCE
# =====================================================

def save_bm25(session_id, bm25):
    index_path = _index_path(session_id)
    _ensure_dir(index_path)

    with open(_bm25_path(session_id), "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(session_id):
    path = _bm25_path(session_id)
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)