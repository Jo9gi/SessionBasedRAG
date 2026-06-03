import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# =====================================================
# DYNAMIC DOCUMENT LOADER (PDF, DOCX, TXT, EXCEL)
# =====================================================

def load_document(file_path):
    """
    Loads and parses a document dynamically based on its file extension.
    Supported extensions: .pdf, .docx, .txt, .xlsx, .xls
    """
    ext = os.path.splitext(file_path)[1].lower()
    documents = []

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif ext == ".docx":
        # Docx2txtLoader uses the docx2txt package to extract text from Word files
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

    elif ext == ".txt":
        # TextLoader reads raw text files with a specific encoding
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

    elif ext in [".xlsx", ".xls"]:
        # Excel files can contain multiple sheets. We parse each sheet and format it as text.
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets as dict of DataFrames
            for sheet_name, df in excel_data.items():
                # Convert the dataframe to a clean CSV-like string representation
                df_string = df.to_csv(index=False)
                doc = Document(
                    page_content=f"Sheet Name: {sheet_name}\n\n{df_string}",
                    metadata={
                        "source": file_path,
                        "sheet_name": sheet_name
                    }
                )
                documents.append(doc)
        except Exception as e:
            raise IOError(f"Failed to read Excel file {file_path}: {str(e)}")

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Standardize metadata keys for all loaded document pages/sheets
    file_name = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = file_path
        doc.metadata["file_name"] = file_name
        doc.metadata["page"] = doc.metadata.get("page", 0)

    return documents
