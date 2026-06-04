import os
import uuid
from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from rest_framework.parsers import MultiPartParser, FormParser

from chroma_rag.models import ChatSession
from .services.document_loader import load_document
from .services.chunking import recursive_chunking
from .services.vectordb import create_vector_store, load_vector_store
from .services.hybrid_search import hybrid_search, build_bm25
from .services.reranker import rerank_chunks
from .services.llm import generate_response, condense_question
from .services.session_store import get_history, add_to_history, clear_history


# =====================================================
# IN-PROCESS CACHE
# =====================================================
# Keeps the most recently used session's vector_store, chunks, and bm25
# in memory so repeated queries on the same session don't reload from disk.
# When the user switches to a different session, the cache is refreshed.
# =====================================================
_rag_cache = {
    "session_id":   None,
    "vector_store": None,
    "chunks":       None,
    "bm25":         None,
}


class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField(help_text="Upload a document (.pdf, .docx, .txt, .xlsx, .xls)")


# =====================================================
# UPLOAD VIEW
# =====================================================
# POST: User uploads a document.
#   - Generates a unique session_id for this upload.
#   - Saves the file into a session-specific folder so old files are never deleted.
#   - Creates a new ChatSession row in the database.
#   - Builds the Chroma vector store for this session only.
#   - Warms the in-process cache so the first query is fast.
#   - Returns the session_id — the frontend must store and send this on every query.
# GET: Returns all past sessions so the UI can show a chat history sidebar.
# =====================================================

class RAGUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = FileUploadSerializer

    def get_serializer(self, *args, **kwargs):
        return FileUploadSerializer(*args, **kwargs)

    def get(self, request, *args, **kwargs):
        # Return all sessions ordered newest first — used by the UI sidebar
        sessions = ChatSession.objects.order_by("-created_at").values(
            "session_id", "document_name", "created_at"
        )
        return Response({"sessions": list(sessions)}, status=status.HTTP_200_OK)

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
            # 1. Generate a unique session_id for this upload
            session_id = str(uuid.uuid4())

            # 2. Save the file into a session-specific folder
            #    Each session has its own folder — old files are never overwritten or deleted
            session_doc_dir = os.path.join("chroma_rag", "documents", session_id)
            os.makedirs(session_doc_dir, exist_ok=True)
            fs = FileSystemStorage(location=session_doc_dir)
            saved_name = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(session_doc_dir, saved_name)

            # 3. Each session gets its own isolated Chroma vector DB folder
            chroma_path = os.path.join("chroma_sessions", session_id)

            # 4. Parse, chunk, and build the vector store for this session
            documents = load_document(file_path)
            chunks = recursive_chunking(documents)
            vector_store = create_vector_store(chunks, chroma_path)

            # 5. Save session metadata to the database
            #    This is what allows us to reload the session later
            ChatSession.objects.create(
                session_id=session_id,
                document_name=saved_name,
                document_path=file_path,
                chroma_path=chroma_path
            )

            # 6. Warm the in-process cache so the first query hits memory, not disk
            _rag_cache["session_id"]   = session_id
            _rag_cache["vector_store"] = vector_store
            _rag_cache["chunks"]       = chunks
            _rag_cache["bm25"]         = build_bm25(chunks)

            return Response(
                {
                    "status":        "success",
                    "message":       "Document uploaded and indexed. Use the session_id for all queries.",
                    "session_id":    session_id,
                    "document_name": saved_name,
                    "total_chunks":  len(chunks)
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# =====================================================
# QUERY VIEW
# =====================================================
# POST: User sends a query with a session_id.
#   - Loads the session from DB to get the document path and chroma path.
#   - Uses in-process cache if the session matches (fast path — same session).
#   - Loads from disk if the user switched to an old session (cold path).
#   - Condenses the question using chat history if history exists.
#   - Retrieves relevant chunks, reranks, generates answer.
#   - Saves the Q&A turn to the database so history persists across restarts.
# =====================================================

class RAGQueryAPIView(APIView):

    def post(self, request, *args, **kwargs):
        original_query = request.data.get("query")
        session_id     = request.data.get("session_id")
        page_filter    = request.data.get("page_filter")

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

            # 2. Load vector_store, chunks, bm25 from cache or disk
            #    Fast path: same session as the last query — everything is already in memory
            #    Cold path: user opened an old session — load its vector DB and document from disk
            if _rag_cache["session_id"] == session_id and _rag_cache["vector_store"] is not None:
                vector_store = _rag_cache["vector_store"]
                chunks       = _rag_cache["chunks"]
                bm25         = _rag_cache["bm25"]
            else:
                vector_store = load_vector_store(session.chroma_path)
                chunks       = recursive_chunking(load_document(session.document_path))
                bm25         = build_bm25(chunks)
                _rag_cache.update({
                    "session_id":   session_id,
                    "vector_store": vector_store,
                    "chunks":       chunks,
                    "bm25":         bm25
                })

            # 3. Load this session's chat history from the database
            chat_history = get_history(session_id)

            # 4. Condense the question using chat history if history exists.
            #    First query in a session — history is empty, use the original query directly.
            #    Follow-up queries — rewrite into a standalone question so retrieval works correctly.
            #    Example: "What about card payments?" → "What is the refund policy for card payments?"
            if chat_history:
                search_query = condense_question(chat_history, original_query)
            else:
                search_query = original_query

            # 5. Handle optional page filter
            filter_expr = None
            if page_filter is not None:
                filter_expr = f'metadata["page"] == {page_filter}'

            # 6. Hybrid Search — BM25 + Dense vector search, pre-built index, no rebuild
            retrieved_docs = hybrid_search(
                query=search_query, vector_store=vector_store,
                chunks=chunks, bm25=bm25, top_k=10, filter=filter_expr
            )

            # 7. Rerank top results using Cross-Encoder to pick the best 5
            reranked_docs = rerank_chunks(query=search_query, chunks=retrieved_docs, top_k=5)

            # 8. Generate answer — history is passed so the LLM gives a conversational response
            answer = generate_response(
                query=original_query, retrieved_docs=reranked_docs, chat_history=chat_history
            )

            # 9. Save this Q&A turn to the database — persists across server restarts
            add_to_history(session_id, original_query, answer)

            return Response(
                {
                    "session_id":    session_id,
                    "document_name": session.document_name,
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
                "document_name": session.document_name,
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


class RAGUIView(View):
    def get(self, request):
        return render(request, "chroma_rag/index.html")
