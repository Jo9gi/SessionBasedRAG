from rank_bm25 import BM25Okapi

def build_bm25_index(chunks):

    tokenized = [doc.page_content.lower().split() for doc in chunks]

    index = BM25Okapi(tokenized)

    return index


def bm25_search(query, index, chunks, top_k=20):

    tokenized_query = query.lower().split()

    scores = index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [chunks[i] for i in top_indices]


# =====================================================
# RECIPROCAL RANK FUSION
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

            scores[doc_id] += 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[doc_id] for doc_id, _ in ranked]


# =====================================================
# HYBRID SEARCH
# =====================================================

def hybrid_search(query, vector_store, chunks, top_k=20, filter=None):

    from .retrieval import retrieve_chunks

    # dense search via Milvus
    dense_results = retrieve_chunks(
        query=query,
        vector_store=vector_store,
        top_k=top_k,
        filter=filter
    )

    # BM25 keyword search
    bm25_index = build_bm25_index(chunks)
    bm25_results = bm25_search(query, bm25_index, chunks, top_k=top_k)

    print(f"\nHYBRID SEARCH:")
    print(f"  Dense results : {len(dense_results)}")
    print(f"  BM25  results : {len(bm25_results)}")

    # merge with RRF
    fused = reciprocal_rank_fusion([dense_results, bm25_results])

    return fused[:top_k]
