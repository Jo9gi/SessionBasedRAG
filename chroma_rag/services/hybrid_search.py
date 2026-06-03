from rank_bm25 import BM25Okapi
from .retrieval import retrieve_chunks

# =====================================================
# RECIPROCAL RANK FUSION (RRF)
# =====================================================

def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    doc_map = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content

            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = doc

            # Add reciprocal rank score to the document
            scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort documents by their combined RRF score in descending order
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[doc_id] for doc_id, _ in ranked]


# =====================================================
# HYBRID SEARCH WITH RRF
# =====================================================

import re

def clean_tokenize(text):
    # Lowercase, replace non-alphanumeric except hyphen with space, split
    cleaned = re.sub(r'[^\w\-]', ' ', text.lower())
    return [t for t in cleaned.split() if t]

def hybrid_search(query, vector_store, chunks, top_k=20, filter=None):
    # 1. Build BM25 Index on the fly
    tokenized_docs = [clean_tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_docs)

    # 2. BM25 Search (Sparse Search)
    query_tokens = clean_tokenize(query)
    bm25_results = bm25.get_top_n(query_tokens, chunks, n=top_k)

    # 3. Dense Search (ChromaDB)
    dense_results = retrieve_chunks(query=query, vector_store=vector_store, top_k=top_k, filter=filter)

    # 4. Merge Results using Reciprocal Rank Fusion (RRF)
    fused_results = reciprocal_rank_fusion([dense_results, bm25_results], k=60)

    print(f"\nHYBRID SEARCH (RRF BLENDED - CHROMA):")
    print(f"  Dense results retrieved : {len(dense_results)}")
    print(f"  BM25 results retrieved  : {len(bm25_results)}")
    print(f"  RRF Blended results     : {len(fused_results)}")

    # 5. Final Results sliced to top_k
    return fused_results[:top_k]
