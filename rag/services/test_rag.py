from .document_loader import load_document
from .chunking import recursive_chunking
from .vectordb import create_vector_store
from .retrieval import retrieve_chunks
from .llm import generate_response
from .query_rewriter import rewrite_query
from .multi_query import multi_query_retrieve
from .reranker import rerank_chunks


# =====================================================
# LOAD PDF
# =====================================================

documents = load_document(

    "rag/documents/test.pdf"
)

print(f"\nLOADED PAGES: {len(documents)}")
print(f"SAMPLE METADATA: {documents[0].metadata}")


# =====================================================
# CHUNKING
# =====================================================

chunks = recursive_chunking(documents)

print(f"\nTOTAL CHUNKS: {len(chunks)}")

vector_store = create_vector_store(
    chunks
)

print("\nMILVUS VECTOR STORE CREATED")


# =====================================================
# CHAT LOOP
# =====================================================

while True:

    query = input("\nAsk Question: ")
    if query.lower() == "exit":
        break

    page_filter = input("Filter by page number (press Enter to skip): ").strip()

    filter_expr = None
    if page_filter.isdigit():
        filter_expr = f"metadata[\"page\"] == {page_filter}"

    rewritten = rewrite_query(query)
    print(f"\nREWRITTEN QUERY: {rewritten}")

    retrieved_docs = multi_query_retrieve(
        query=rewritten,
        vector_store=vector_store,
        chunks=chunks,
        top_k=20,
        filter=filter_expr
    )

    reranked_docs = rerank_chunks(query=rewritten, chunks=retrieved_docs, top_k=5)

    print("\nRETRIEVED CHUNKS:\n")

    for doc in reranked_docs:

        print(f"[page={doc.metadata.get('page')} | file={doc.metadata.get('file_name')}]")
        print(doc.page_content)
        print("\n" + "=" * 50)

    answer = generate_response(query=query, retrieved_docs=reranked_docs)

    print("\nFINAL ANSWER:\n")
    print(answer)