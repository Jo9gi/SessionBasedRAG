from .document_loader import load_document
from .chunking import recursive_chunking
from .vectordb import create_vector_store
from .retrieval import retrieve_chunks
from .llm import generate_response
from .query_rewriter import rewrite_query
from .hybrid_search import hybrid_search
from .reranker import rerank_chunks

# =====================================================
# LOAD PDF (Local app directory)
# =====================================================

pdf_path = "chroma_rag/documents/test.pdf"
documents = load_document(pdf_path)

print(f"\nLOADED PAGES: {len(documents)}")
print(f"SAMPLE METADATA: {documents[0].metadata}")


# =====================================================
# CHUNKING
# =====================================================

chunks = recursive_chunking(documents)

print(f"\nTOTAL CHUNKS: {len(chunks)}")

# =====================================================
# CREATE CHROMA VECTOR STORE (Zero-Docker local storage!)
# =====================================================

vector_store = create_vector_store(chunks)

print("\nCHROMA VECTOR STORE CREATED (LOCAL FILE PERSISTED)")


# =====================================================
# CHAT LOOP
# =====================================================

while True:
    original_query = input("\nAsk Question (Chroma RAG): ")
    if original_query.lower() == "exit":
        break

    page_filter = input("Filter by page number (press Enter to skip): ").strip()

    filter_expr = None
    if page_filter.isdigit():
        filter_expr = f"metadata[\"page\"] == {page_filter}"

    # 1. Rewrite Query
    rewritten = rewrite_query(original_query)
    print(f"\nREWRITTEN QUERY: {rewritten}")

    # 2. Hybrid Search (RRF Blended Dense + Sparse)
    retrieved_docs = hybrid_search(
        query=rewritten, 
        vector_store=vector_store, 
        chunks=chunks, 
        top_k=20, 
        filter=filter_expr
    )

    # 3. Rerank Chunks using Cross-Encoder
    reranked_docs = rerank_chunks(query=rewritten, chunks=retrieved_docs, top_k=5)

    print("\nRETRIEVED CHUNKS:\n")
    for doc in reranked_docs:
        print(f"[page={doc.metadata.get('page')} | file={doc.metadata.get('file_name')}]")
        print(doc.page_content)
        print("\n" + "=" * 50)

    # 4. Generate Final Answer grounded in context
    answer = generate_response(query=original_query, retrieved_docs=reranked_docs)

    print("\nFINAL ANSWER:\n")
    print(answer)
