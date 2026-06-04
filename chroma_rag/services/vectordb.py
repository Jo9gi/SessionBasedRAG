import os
import shutil
from langchain_chroma import Chroma
from .embedding import embedding_model


# =====================================================
# CREATE CHROMA VECTOR STORE (PER SESSION)
# =====================================================
# Each session gets its own chroma_path (e.g. ./chroma_sessions/<session_id>).
# This means every uploaded document has its own isolated vector database.
# Old sessions keep their vectors intact — they are never overwritten.
# =====================================================

def create_vector_store(chunks, chroma_path):
    # Clear existing DB at this path if it exists (re-index case)
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_path
    )
    return vector_store


def load_vector_store(chroma_path):
    # =====================================================
    # LOAD EXISTING CHROMA VECTOR STORE
    # =====================================================
    # Called when a user opens an old session.
    # Loads the vector DB that was created during the original upload.
    # =====================================================
    return Chroma(persist_directory=chroma_path, embedding_function=embedding_model)
