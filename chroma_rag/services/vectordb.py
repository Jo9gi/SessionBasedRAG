import os
import shutil
from langchain_chroma import Chroma
from .embedding import embedding_model


def create_vector_store(chunks, chroma_path):
    batch_size = 100
    if os.path.exists(chroma_path):
        vector_store = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)
        for i in range(0, len(chunks), batch_size):
            vector_store.add_documents(chunks[i:i + batch_size])
    else:
        # Initialize with the first batch
        vector_store = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=embedding_model,
            persist_directory=chroma_path
        )
        # Add remaining batches
        for i in range(batch_size, len(chunks), batch_size):
            vector_store.add_documents(chunks[i:i + batch_size])
            
    return vector_store


def load_vector_store(chroma_path):

    return Chroma(persist_directory=chroma_path, embedding_function=embedding_model)
