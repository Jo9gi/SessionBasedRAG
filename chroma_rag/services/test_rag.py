import os
import sys
import uuid

# Force UTF-8 output so Windows cp1252 terminal doesn't crash on special chars from PDFs
sys.stdout.reconfigure(encoding='utf-8')

# Make sure project root (SessionBasedRAG/) is on sys.path so all imports work
# This file lives at: chroma_rag/services/test_rag.py
# We go up two levels:  services/ -> chroma_rag/ -> SessionBasedRAG/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── Import exactly the same services views.py uses ──────────────────────────
from chroma_rag.services.document_loader import load_document
from chroma_rag.services.chunking import recursive_chunking
from chroma_rag.services.vectordb import create_vector_store
from chroma_rag.services.hybrid_search import build_bm25, hybrid_search
from chroma_rag.services.reranker import rerank_chunks
from chroma_rag.services.llm import generate_response, condense_question

# NOTE: query_rewriter is NOT used here — the real app (views.py) does NOT use it either.
# views.py uses condense_question() only when chat_history exists (follow-up questions).
# For the first question in a session, the raw query is used directly — same as here.

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

PDF_PATH = os.path.join(
    BASE_DIR,
    "chroma_rag", "documents",
    "aec28638-d863-4819-a023-2a8ddf9ba0e1",
    "MachineLearning.pdf"
)

# Unique path per run — mirrors how views.py creates chroma_sessions/<session_id>/
# This avoids Windows file-lock errors when re-running while previous ChromaDB is still open.
TEST_CHROMA_PATH = os.path.join(BASE_DIR, "chroma_test_temp", str(uuid.uuid4()))

# =============================================================================
# STEP 1: LOAD + CHUNK DOCUMENT
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: LOADING DOCUMENT")
print("  File : " + os.path.basename(PDF_PATH))
print("=" * 60)

documents = load_document(PDF_PATH)

print("[OK] Pages loaded   : " + str(len(documents)))
print("[OK] Sample metadata: " + str(documents[0].metadata))

# --- CHUNKING ---
print("\n" + "-" * 60)
print("CHUNKING  (chunk_size=1000, overlap=200)")
print("-" * 60)

chunks = recursive_chunking(documents)

print("[OK] Total chunks   : " + str(len(chunks)))

# Show first 3 chunk previews — use this to verify chunking quality
print("\n--- CHUNK PREVIEW (first 3) ---")
for i, chunk in enumerate(chunks[:3]):
    print(
        "\n  [Chunk " + str(i + 1) + "]"
        " page=" + str(chunk.metadata.get("page")) +
        " | chars=" + str(len(chunk.page_content))
    )
    print("  " + chunk.page_content[:300])
    print("  ...")

# =============================================================================
# STEP 2: BUILD VECTOR STORE + BM25
# (mirrors what views.py does during file upload)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: BUILDING VECTOR STORE + BM25 INDEX")
print("  Chroma path: " + TEST_CHROMA_PATH)
print("=" * 60)

vector_store = create_vector_store(chunks, chroma_path=TEST_CHROMA_PATH)
print("[OK] Chroma vector store created")

bm25 = build_bm25(chunks)
print("[OK] BM25 index built")

# =============================================================================
# STEP 3: CHAT LOOP
# This is a LOCAL simulation of the full views.py RAGQueryAPIView flow:
#
#   views.py flow per query:
#   ─────────────────────────────────────────────────────────
#   1. Get chat_history (from DB via session_store.get_history)
#   2. If history exists  → condense_question(history, query)
#      If first question  → use raw query directly
#   3. Hybrid search      → BM25 + Dense, top_k=10
#   4. Rerank             → Cross-Encoder, top_k=8
#   5. Generate answer    → grounded in context + history
#   6. Save to history    → add_to_history(session_id, q, a)
#   ─────────────────────────────────────────────────────────
#   Here we store chat_history in a plain Python list (no DB needed for testing).
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: CHAT LOOP  (mirrors views.py RAGQueryAPIView exactly)")
print("  - First question  : raw query used directly (no condensing)")
print("  - Follow-up questions: condense_question() rewrites using history")
print("  Type 'exit' to quit | 'history' to view chat history | 'clear' to reset history")
print("=" * 60)

# Local in-memory chat history — simulates the DB-backed session_store
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
chat_history = []

while True:
    original_query = input("\nYou: ").strip()

    if original_query.lower() == "exit":
        print("\nGoodbye!")
        break

    if original_query.lower() == "history":
        if not chat_history:
            print("  [No history yet]")
        else:
            print("\n--- CHAT HISTORY ---")
            for msg in chat_history:
                role = "You" if msg["role"] == "user" else "Bot"
                print("  " + role + ": " + msg["content"])
        continue

    if original_query.lower() == "clear":
        chat_history = []
        print("  [History cleared]")
        continue

    print("\n" + "-" * 60)

    # ── STEP A: Condense question (mirrors views.py) ──────────────────────────
    # Use only the last HISTORY_WINDOW turns — same as views.py
    # Full history is kept locally for the 'history' command, but only the
    # recent window is sent to the LLM to avoid growing latency.
    HISTORY_WINDOW = 3
    recent_history = chat_history[-(HISTORY_WINDOW * 2):] if chat_history else []

    if recent_history:
        total_turns = len(chat_history) // 2
        window_turns = len(recent_history) // 2
        print("[CONDENSE] History: " + str(total_turns) + " total turn(s), using last " + str(window_turns) + " for context ...")
        search_query = condense_question(recent_history, original_query)
        print("  Original  : " + original_query)
        print("  Condensed : " + search_query)
    else:
        print("[CONDENSE] No history yet — using raw query for retrieval")
        search_query = original_query
        print("  Search    : " + search_query)

    # ── STEP B: Hybrid Search (mirrors views.py) ─────────────────────────────
    # top_k=20: wide net so reranker has enough high-quality candidates
    print("\n[SEARCH] Hybrid search  (BM25 + Dense ChromaDB, top_k=20) ...")
    retrieved_docs = hybrid_search(query=search_query, vector_store=vector_store, chunks=chunks, bm25=bm25, top_k=20)
    print("  Retrieved " + str(len(retrieved_docs)) + " candidate chunks")

    if not retrieved_docs:
        print("\n[WARNING] No chunks retrieved! This is why you get 'I could not find relevant information.'")
        print("  Possible causes:")
        print("    - Query too vague or different from document vocabulary")
        print("    - Chunk size too large / content not indexed properly")
        print("-" * 60)
        continue

    # ── STEP C: Rerank (mirrors views.py) ──────────────────────────────────
    # top_k=5: tight quality filter — only the best 5 (above MIN_SCORE) reach the LLM
    print("\n[RERANK] Cross-Encoder reranking (top_k=5) ...")
    reranked_docs = rerank_chunks(query=search_query, chunks=retrieved_docs, top_k=6)

    # Show each chunk so you can debug what the LLM sees
    print("\n--- CHUNKS SENT TO LLM ---")
    for idx, doc in enumerate(reranked_docs, 1):
        safe_content = doc.page_content[:500].encode('utf-8', errors='replace').decode('utf-8')
        print(
            "\n  [" + str(idx) + "] page=" + str(doc.metadata.get("page")) +
            " | chars=" + str(len(doc.page_content))
        )
        print("  " + safe_content)
        print("  " + "." * 56)

    # ── STEP D: Generate Answer (mirrors views.py) ──────────────────────────────────
    # Passes original_query (not condensed) and only recent_history — same as views.py
    print("\n[GENERATE] Calling LLM ...")
    answer = generate_response(
        query=original_query,
        retrieved_docs=reranked_docs,
        chat_history=recent_history
    )

    print("\n" + "=" * 60)
    print("Bot: " + answer)
    print("=" * 60)

    # ── STEP E: Save to local history (mirrors add_to_history in views.py) ─
    chat_history.append({"role": "user",      "content": original_query})
    chat_history.append({"role": "assistant", "content": answer})

    print("  [History updated: " + str(len(chat_history) // 2) + " turn(s) stored]")
