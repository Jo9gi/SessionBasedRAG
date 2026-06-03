import os
import shutil
from langchain_chroma import Chroma
from .embedding import embedding_model

# =====================================================
# CREATE CHROMA VECTOR STORE
# =====================================================

def create_vector_store(chunks):
    persist_directory = "./chroma_db"

    # Replicate drop_old=True behavior by clearing previous database files if they exist
    if os.path.exists(persist_directory):
        print(f"\nClearing existing Chroma database at {persist_directory}...")
        shutil.rmtree(persist_directory)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    # Close the client connection to release locks on Windows
    if hasattr(vector_store, "_client") and hasattr(vector_store._client, "close"):
        vector_store._client.close()

    return vector_store

