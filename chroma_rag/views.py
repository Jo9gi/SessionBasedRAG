import os
import shutil
from django.core.files.storage import FileSystemStorage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain_chroma import Chroma

# Import custom RAG services
from .services.document_loader import load_document
from .services.chunking import recursive_chunking
from .services.embedding import embedding_model
from .services.vectordb import create_vector_store
from .services.query_rewriter import rewrite_query
from .services.hybrid_search import hybrid_search
from .services.reranker import rerank_chunks
from .services.llm import generate_response

from rest_framework import serializers
from rest_framework.parsers import MultiPartParser, FormParser

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField(help_text="Upload a document (.pdf, .docx, .txt, .xlsx, .xls)")

class RAGUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = FileUploadSerializer

    def get_serializer(self, *args, **kwargs):
        return FileUploadSerializer(*args, **kwargs)

    def get(self, request, *args, **kwargs):
        documents_dir = "chroma_rag/documents"
        active_file_name = None
        if os.path.exists(documents_dir) and os.listdir(documents_dir):
            active_file_name = os.listdir(documents_dir)[0]
        return Response(
            {
                "active_document": active_file_name,
                "supported_extensions": ['.pdf', '.docx', '.txt', '.xlsx', '.xls']
            }, 
            status=status.HTTP_200_OK
        )

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get('file')

        # 1. Validate File Presence
        if not uploaded_file:
            return Response(
                {"error": "No file was uploaded. Please send a 'file' parameter with the document binary in a multipart request."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Validate File Extension
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        allowed_extensions = ['.pdf', '.docx', '.txt', '.xlsx', '.xls']
        if ext not in allowed_extensions:
            return Response(
                {"error": f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(allowed_extensions)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        documents_dir = "chroma_rag/documents"

        try:
            # 3. Clear previous documents in the documents folder (Enforce Single-Doc RAG)
            if os.path.exists(documents_dir):
                for filename in os.listdir(documents_dir):
                    file_path = os.path.join(documents_dir, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
            else:
                os.makedirs(documents_dir)

            # 4. Save the new file locally
            fs = FileSystemStorage(location=documents_dir)
            saved_name = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(documents_dir, saved_name)

            # 5. Parse, Chunk, and Embed the file into ChromaDB
            documents = load_document(file_path)
            chunks = recursive_chunking(documents)
            create_vector_store(chunks)

            return Response(
                {
                    "status": "success",
                    "message": "Document successfully uploaded and indexed.",
                    "file_name": saved_name,
                    "file_type": ext,
                    "total_chunks": len(chunks)
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {"error": f"An error occurred during file ingestion: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RAGQueryAPIView(APIView):

    def post(self, request, *args, **kwargs):
        original_query = request.data.get("query")
        page_filter = request.data.get("page_filter")

        if not original_query:
            return Response(
                {"error": "The 'query' parameter is required in the request body."}, status=status.HTTP_400_BAD_REQUEST)

        documents_dir = "chroma_rag/documents"

        # 1. Detect the currently active document
        if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
            return Response(
                {
                    "error": "No active document found in the RAG system.",
                    "suggestion": "Please upload a document first: POST /api/chroma/upload/"
                },
                status=status.HTTP_404_NOT_FOUND
            )

        active_files = os.listdir(documents_dir)
        active_file_name = active_files[0]
        file_path = os.path.join(documents_dir, active_file_name)
        persist_directory = "./chroma_db"

        # 2. Verify Database exists
        if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
            return Response(
                {
                    "error": "Chroma vector database not found or empty.",
                    "suggestion": "Please trigger the indexing endpoint or upload a file first."
                },
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            # 3. Load the persisted Chroma vector store
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

            # 4. Handle Page Filtering
            filter_expr = None
            if page_filter is not None:
                filter_expr = f"metadata[\"page\"] == {page_filter}"

            # 5. Parse the active file on-the-fly for the local BM25 keyword search index
            documents = load_document(file_path)
            chunks = recursive_chunking(documents)

            # 6. Domain-Adaptive Query Rewriting (Infers domain dynamically using the active file name)
            rewritten = rewrite_query(original_query, file_context=active_file_name)

            # 7. Execute Hybrid Search (Dense Chroma + Sparse BM25 RRF Blended)
            retrieved_docs = hybrid_search(query=rewritten, vector_store=vector_store, chunks=chunks, top_k=25, filter=filter_expr)

            # Close Chroma database client to release locks on Windows
            if hasattr(vector_store, "_client") and hasattr(vector_store._client, "close"):
                vector_store._client.close()

            # 8. Rerank using Cross-Encoder
            reranked_docs = rerank_chunks(query=rewritten, chunks=retrieved_docs, top_k=5)

            # 9. Generate Final Response
            answer = generate_response(query=original_query, retrieved_docs=reranked_docs)

            # 10. Format Response
            sources = []
            for doc in reranked_docs:
                sources.append(
                    {
                        "page": doc.metadata.get("page"),
                        "file_name": doc.metadata.get("file_name"),
                        "content": doc.page_content
                    }
                )

            response_payload = {
                "active_document": active_file_name,
                "original_query": original_query,
                # "rewritten_query": rewritten,
                "answer": answer,
                # "sources": sources
            }

            return Response(response_payload, status=status.HTTP_200_OK)

        except Exception as e:
            # Clean up Chroma client on error if instantiated
            if "vector_store" in locals():
                if hasattr(vector_store, "_client") and hasattr(vector_store._client, "close"):
                    try:
                        vector_store._client.close()
                    except:
                        pass
            return Response(
                {"error": f"An unexpected error occurred during search: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RAGIndexAPIView(APIView):
    def post(self, request, *args, **kwargs):
        documents_dir = "chroma_rag/documents"
        
        if not os.path.exists(documents_dir) or not os.listdir(documents_dir):
            return Response({"error": "No documents found in the active folder to index. Please upload one first."}, status=status.HTTP_404_NOT_FOUND)

        active_files = os.listdir(documents_dir)
        active_file_name = active_files[0]
        file_path = os.path.join(documents_dir, active_file_name)

        try:
            documents = load_document(file_path)
            chunks = recursive_chunking(documents)
            create_vector_store(chunks)

            return Response(
                {
                    "status": "success",
                    "message": "Chroma vector store successfully created and persisted.",
                    "file_indexed": active_file_name,
                    "total_chunks": len(chunks)
                },
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {"error": f"An error occurred during indexing: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
