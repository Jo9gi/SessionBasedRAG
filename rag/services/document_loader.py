import os
from langchain_community.document_loaders import PyPDFLoader


# =====================================================
# LOAD DOCUMENT
# =====================================================

def load_document(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    file_name = os.path.basename(file_path)

    for doc in documents:

        doc.metadata["source"] = file_path
        doc.metadata["file_name"] = file_name
        doc.metadata["page"] = doc.metadata.get("page", 0)

    return documents