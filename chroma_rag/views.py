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
from chroma_rag.models import ChatSession, SessionDocument, ChatMessage
from .services.document_loader import load_document
from .services.chunking import recursive_chunking
from .services.vectordb import create_vector_store, load_vector_store
from .services.hybrid_search import hybrid_search, build_bm25
from .services.reranker import rerank_chunks
from .services.llm import generate_response, condense_question, llm
from .services.session_store import get_history, add_to_history, clear_history

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
                chroma_path = session.chroma_path
            else:
                # Create new session
                session_id = str(uuid.uuid4())
                session_doc_dir = os.path.join("chroma_rag", "documents", session_id)
                chroma_path = os.path.join("chroma_sessions", session_id)
                
                # Generate session name using LLM
                prompt = f"Generate a short, concise 3-5 word title for a document named '{uploaded_file.name}'."
                try:
                    session_name = llm.invoke(prompt).content.strip().replace('"', '')
                except:
                    session_name = uploaded_file.name
                    
                session = ChatSession.objects.create(
                    session_id=session_id,
                    session_name=session_name,
                    chroma_path=chroma_path
                )

            os.makedirs(session_doc_dir, exist_ok=True)
            fs = FileSystemStorage(location=session_doc_dir)
            saved_name = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(session_doc_dir, saved_name)

            sdoc = SessionDocument.objects.create(
                session=session,
                document_name=saved_name,
                document_path=file_path
            )

            documents = load_document(file_path)
            
            for doc in documents:
                doc.metadata["file_name"] = saved_name
                doc.metadata["doc_id"] = str(sdoc.id)
                doc.metadata["session_id"] = session.session_id
                
            chunks = recursive_chunking(documents)
            
            vector_store = create_vector_store(chunks, chroma_path)

            try:
                stored_data = vector_store.get()
                all_chunks = [Document(page_content=doc_text, metadata=meta) for doc_text, meta in zip(stored_data.get("documents", []), stored_data.get("metadatas", []))]
            except Exception as e:
                print(f"Error fetching existing chunks: {e}")
                all_chunks = chunks

            _rag_cache["session_id"]   = session_id
            _rag_cache["vector_store"] = vector_store
            _rag_cache["chunks"]       = all_chunks
            _rag_cache["bm25"]         = build_bm25(all_chunks)

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
        session_id     = request.data.get("session_id")

        if not original_query:
            return Response({"error": "'query' is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not session_id:
            return Response(
                {"error": "'session_id' is required. Upload a document first to get one."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 1. Load session from DB — confirms the session exists and gives us file + chroma paths
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
                vector_store = load_vector_store(session.chroma_path)
                
                # Fetch chunks directly from ChromaDB for instant, offline load
                try:
                    stored_data = vector_store.get()
                    
                    chunks = [Document(page_content=doc_text, metadata=meta) for doc_text, meta in zip(stored_data.get("documents", []), stored_data.get("metadatas", []))]
                except Exception as e:
                    print(f"Error loading chunks directly from Chroma: {e}")
                    chunks = []

                # Fallback to parsing the original documents if Chroma retrieval failed or returned empty
                if not chunks:
                    try:
                        all_docs_chunks = []
                        for sdoc in session.documents.all():
                            if os.path.exists(sdoc.document_path):
                                all_docs_chunks.extend(recursive_chunking(load_document(sdoc.document_path)))
                            else:
                                print(f"Document file does not exist at {sdoc.document_path}")
                        chunks = all_docs_chunks
                    except Exception as e:
                        print(f"Error loading fallback documents: {e}")
                        chunks = []

                if chunks:
                    bm25 = build_bm25(chunks)
                else:
                    bm25 = None

                _rag_cache.update({
                    "session_id":   session_id,
                    "vector_store": vector_store,
                    "chunks":       chunks,
                    "bm25":         bm25
                })

            # 3. Load this session's chat history from the database
            chat_history = get_history(session_id)

        
            HISTORY_WINDOW = 3   # number of recent Q+A turns to keep (each turn = 2 messages)
            recent_history = chat_history[-(HISTORY_WINDOW * 2):] if chat_history else []

            if recent_history:
                search_query = condense_question(recent_history, original_query)
            else:
                search_query = original_query

            if chunks and bm25 is not None:
                retrieved_docs = hybrid_search(
                    query=search_query, vector_store=vector_store,
                    chunks=chunks, bm25=bm25, top_k=20
                )
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

            # Delete ChromaDB vector store directory from disk
            if os.path.exists(session.chroma_path):
                try:
                    shutil.rmtree(session.chroma_path)
                except Exception as e:
                    print(f"Error removing chroma path: {e}")

            # Delete uploaded document from disk
            if os.path.exists(session.document_path):
                try:
                    os.remove(session.document_path)
                    # Clean up parent directory if empty
                    parent_dir = os.path.dirname(session.document_path)
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
