import os
import uuid
from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from rest_framework.parsers import MultiPartParser, FormParser
from langchain_core.documents import Document
from chroma_rag.models import ChatSession, SessionDocument, ChatMessage, DocumentMetadata
from .services.document_loader import load_document
from .services.chunking import chapter_recursive_chunking
from .services.vectordb import (create_vector_store, load_vector_store, save_chunks, load_chunks, save_bm25, load_bm25)
from .services.hybrid_search import hybrid_search, build_bm25
from .services.reranker import rerank_chunks
from .services.llm import generate_response, condense_question, llm
from .services.session_store import get_history, add_to_history, clear_history
from .services.cache import LRUCache
from .services.query_router import detect_query_type
from .services.metadata_extractor import extract_headings

# Global in‑process LRU cache for FAISS stores (capacity 3 sessions)
_faiss_cache = LRUCache(capacity=3)

_rag_cache = {
    "session_id":   None,
    "vector_store": None,
    "chunks":       None,
    "bm25":         None,
            }


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField(help_text="Upload a document (.pdf, .docx, .txt, .xlsx, .xls)")


class RAGUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = FileUploadSerializer

    def get_serializer(self, *args, **kwargs):
        return FileUploadSerializer(*args, **kwargs)

    def get(self, request, *args, **kwargs):
        # Return all sessions ordered newest first — used by the UI sidebar
        sessions = ChatSession.objects.order_by("-created_at")
        session_list = []
        for s in sessions:
            docs = list(s.documents.values_list("document_name", flat=True))
            session_list.append({
                "session_id": s.session_id,
                "session_name": s.session_name,
                "document_names": docs,
                "created_at": s.created_at
            })
        return Response({"sessions": session_list}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        ext = os.path.splitext(uploaded_file.name)[1].lower()
        allowed_extensions = [".pdf", ".docx", ".txt", ".xlsx", ".xls"]
        if ext not in allowed_extensions:
            return Response(
                {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_extensions)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            session_id = request.data.get("session_id")
            
            if session_id:
                # Append to existing session
                try:
                    session = ChatSession.objects.get(session_id=session_id)
                except ChatSession.DoesNotExist:
                    return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
                session_doc_dir = os.path.join("chroma_rag", "documents", session_id)
                # The FAISS store path is stored in session.chroma_path (legacy name)
                # No extra variable needed; we will use session.chroma_path later
            else:
                # Create new session
                session_id = str(uuid.uuid4())
                session_doc_dir = os.path.join("chroma_rag", "documents", session_id)

                # Generate session name using LLM
                prompt = f"Generate a short, concise 3-5 word title for a document named '{uploaded_file.name}'."
                try:
                    session_name = llm.invoke(prompt).content.strip().replace('"', '')
                except:
                    session_name = uploaded_file.name
                    
                # Create a new FAISS storage directory for this session (saved in DB as chroma_path for legacy compatibility)
                chroma_path = os.path.join("faiss_indices", f"{session_id}.index")
                session = ChatSession.objects.create(
                    session_id=session_id,
                    session_name=session_name,
                    chroma_path=chroma_path
                )
                # No DocumentMetadata needed at session creation; will be added per document below

            os.makedirs(session_doc_dir, exist_ok=True)
            fs = FileSystemStorage(location=session_doc_dir)
            saved_name = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(session_doc_dir, saved_name)

            sdoc = SessionDocument.objects.create(
                session=session,
                document_name=saved_name,
                document_path=file_path
            )
            # Store metadata entry for this uploaded document
            # DocumentMetadata.objects.create(
            #     session=session,
            #     document=sdoc,
            #     title=saved_name,
            #     headings="[]"  # empty JSON array placeholder
            # )

            documents = load_document(file_path)
            page_count = len(documents)
            import json

            headings = extract_headings(documents)

            DocumentMetadata.objects.create(
                session=session,
                document=sdoc,
                title=saved_name,
                page_count=page_count,
                headings=json.dumps(headings)
            )

            for doc in documents:
                doc.metadata["file_name"] = saved_name
                doc.metadata["doc_id"] = str(sdoc.id)
                doc.metadata["session_id"] = session.session_id
                
            chunks = chapter_recursive_chunking(documents)
            save_chunks(session_id,chunks)
            vector_store = create_vector_store(chunks, session_id)

            try:
                # FAISS does not expose a get() method; we already have the chunks list.
                all_chunks = chunks
            except Exception as e:
                print(f"Error fetching existing chunks: {e}")
                all_chunks = chunks

            _rag_cache["session_id"]   = session_id
            _rag_cache["vector_store"] = vector_store
            _rag_cache["chunks"]       = all_chunks
            bm25_index = build_bm25(all_chunks)
            save_bm25(session_id,bm25_index)
            _rag_cache["bm25"] = bm25_index

            all_doc_names = list(session.documents.values_list("document_name", flat=True))

            return Response(
                {
                    "status":        "success",
                    "message":       "Document uploaded and indexed. Use the session_id for all queries.",
                    "session_id":    session_id,
                    "session_name":  session.session_name,
                    "document_names": all_doc_names,
                    "total_chunks":  len(all_chunks)
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGQueryAPIView(APIView):
    def post(self, request, *args, **kwargs):
        original_query = request.data.get("query")
        query_lower = original_query.lower()
        session_id = request.data.get("session_id")

        if not original_query:
            return Response({"error": "'query' is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not session_id:
            return Response(
                {"error": "'session_id' is required. Upload a document first to get one."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 1. Load session from DB — confirms the session exists and gives us file paths
            try:
                session = ChatSession.objects.get(session_id=session_id)
            except ChatSession.DoesNotExist:
                return Response(
                    {"error": f"Session '{session_id}' not found. Please upload a document to start a new session."},
                    status=status.HTTP_404_NOT_FOUND
                )

        
            if _rag_cache["session_id"] == session_id and _rag_cache["vector_store"] is not None:
                vector_store = _rag_cache["vector_store"]
                chunks       = _rag_cache["chunks"]
                bm25         = _rag_cache["bm25"]
            else:
                vector_store = _faiss_cache.get(session_id)
                if vector_store is None:
                    vector_store = load_vector_store(session_id)
                    _faiss_cache.put(session_id, vector_store)

                chunks = load_chunks(session_id)
                bm25 = load_bm25(session_id)
                if chunks is None:
                    print(f"[WARNING] chunks.pkl missing for {session_id}")
                    chunks = []

                if bm25 is None:
                    print(f"[WARNING] bm25.pkl missing for {session_id}")
                    bm25 = build_bm25(chunks) if chunks else None


                _rag_cache.update({
                    "session_id":   session_id,
                    "vector_store": vector_store,
                    "chunks":       chunks,
                    "bm25":         bm25
                })
                # Also store the FAISS store in the LRU cache for quick reuse
                _faiss_cache.put(session_id, vector_store)

            # 3. Load this session's chat history from the database
            chat_history = get_history(session_id)

        
            HISTORY_WINDOW = 3   # number of recent Q+A turns to keep (each turn = 2 messages)
            recent_history = chat_history[-(HISTORY_WINDOW * 2):] if chat_history else []

            if recent_history:
                search_query = condense_question(recent_history, original_query)
            else:
                search_query = original_query

            if chunks and bm25 is not None:
                query_type = detect_query_type(original_query)
                if query_type == "semantic":
                    retrieved_docs = hybrid_search(
                        query=search_query, vector_store=vector_store,
                        chunks=chunks, bm25=bm25, top_k=20
                    )
                else:
                    retrieved_docs = []
            else:
                retrieved_docs = []

            # 7. Rerank with Cross-Encoder — filter noise, keep top 5 for LLM
            reranked_docs = rerank_chunks(query=search_query, chunks=retrieved_docs, top_k=5)

            # 8. Generate answer — only recent history is passed to keep context tight
            answer = generate_response(
                query=original_query, retrieved_docs=reranked_docs, chat_history=recent_history
            )

            # 9. Save this Q&A turn to the database — persists across server restarts
            add_to_history(session_id, original_query, answer)

            return Response(
                {
                    "session_id":    session_id,
                    "session_name":  session.session_name,
                    "document_names": list(session.documents.values_list("document_name", flat=True)),
                    "query":         original_query,
                    "answer":        answer,
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# =====================================================
# SESSION HISTORY VIEW
# =====================================================
# GET: Returns the full chat history for a given session.
#      The UI calls this when a user clicks on an old chat in the sidebar.
# =====================================================

class RAGSessionHistoryAPIView(APIView):
    def get(self, request, session_id, *args, **kwargs):
        try:
            session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

        messages = session.messages.values("role", "content", "created_at")
        return Response(
            {
                "session_id":    session_id,
                "session_name":  session.session_name,
                "document_names": list(session.documents.values_list("document_name", flat=True)),
                "messages":      list(messages)
            },
            status=status.HTTP_200_OK
        )


# =====================================================
# CLEAR SESSION VIEW
# =====================================================
# POST: Deletes all chat messages for a session (keeps the document + vector DB intact).
#       The user can still ask new questions — it just resets the conversation.
# =====================================================

class RAGClearSessionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        session_id = request.data.get("session_id")
        if not session_id:
            return Response({"error": "'session_id' is required."}, status=status.HTTP_400_BAD_REQUEST)
        clear_history(session_id)
        return Response(
            {"status": "success", "message": f"Chat history cleared for session '{session_id}'."},
            status=status.HTTP_200_OK
        )


import shutil


class RAGDeleteSessionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        session_id = request.data.get("session_id")
        if not session_id:
            return Response({"error": "'session_id' is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = ChatSession.objects.get(session_id=session_id)

            # Delete FAISS vector store directory from disk (contains index, chunks, bm25)
            faiss_dir = session.chroma_path
            if os.path.isdir(faiss_dir):
                try:
                    shutil.rmtree(faiss_dir)
                except Exception as e:
                    print(f"Error removing FAISS directory: {e}")
            # Delete uploaded document files belonging to this session (stored via SessionDocument entries)
            for sdoc in session.documents.all():
                if os.path.isfile(sdoc.document_path):
                    try:
                        os.remove(sdoc.document_path)
                    except Exception as e:
                        print(f"Error removing document {sdoc.document_path}: {e}")
            # Remove the now‑empty parent folder if it exists
            parent_dir = os.path.join("chroma_rag", "documents", session_id)
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                try:
                    os.rmdir(parent_dir)
                except Exception as e:
                    print(f"Error removing document folder {parent_dir}: {e}")

            # No need to delete session.document_path (does not exist); DB rows are removed later
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                except Exception as e:
                    print(f"Error removing document file/folder: {e}")

            # Delete session from database (cascades to ChatMessage)
            session.delete()

            # Evict from in-process cache if this was the active session
            global _rag_cache
            if _rag_cache["session_id"] == session_id:
                _rag_cache.update({
                    "session_id":   None,
                    "vector_store": None,
                    "chunks":       None,
                    "bm25":         None
                })

            return Response(
                {"status": "success", "message": f"Session '{session_id}' and all associated files deleted successfully."},
                status=status.HTTP_200_OK
            )

        except ChatSession.DoesNotExist:
            return Response({"error": f"Session '{session_id}' not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class RAGUIView(View):
    def get(self, request):
        return render(request, "chroma_rag/index.html")


# =====================================================
# METADATA ENDPOINTS
# =====================================================

class DocumentCountAPIView(APIView):
    """Return total number of uploaded documents across all sessions."""
    def get(self, request, *args, **kwargs):
        total = SessionDocument.objects.count()
        return Response({"total_documents": total}, status=status.HTTP_200_OK)

class TOCAPIView(APIView):
    """Return the table of contents (headings) for a specific document.
    Query parameters: ?session_id=...&document_name=...
    """
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get("session_id")
        document_name = request.query_params.get("document_name")
        if not session_id or not document_name:
            return Response({"error": "session_id and document_name are required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            sdoc = SessionDocument.objects.get(session__session_id=session_id, document_name=document_name)
            meta = DocumentMetadata.objects.get(document=sdoc)
            # headings stored as JSON string; parse safely
            import json
            headings = json.loads(meta.headings) if meta.headings else []
        except (SessionDocument.DoesNotExist, DocumentMetadata.DoesNotExist) as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        return Response({"document": document_name, "headings": headings}, status=status.HTTP_200_OK)
