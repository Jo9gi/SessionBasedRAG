import re
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from .retrieval import retrieve_chunks


def clean_tokenize(text):
    cleaned = re.sub(r'[^\w\-]', ' ', text.lower())
    return [t for t in cleaned.split() if t]


def build_bm25(chunks):
    """Build BM25 index once after upload — pass result into hybrid_search."""
    tokenized_docs = [clean_tokenize(doc.page_content) for doc in chunks]
    return BM25Okapi(tokenized_docs)


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


def hybrid_search(query, vector_store, chunks, bm25, top_k=10, filter=None):
    """BM25 + dense retrieval run in parallel, then fused via RRF."""
    query_tokens = clean_tokenize(query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(bm25.get_top_n, query_tokens, chunks, top_k)
        dense_future = executor.submit(retrieve_chunks, query, vector_store, top_k, filter)
        bm25_results = bm25_future.result()
        dense_results = dense_future.result()

    fused_results = reciprocal_rank_fusion([dense_results, bm25_results], k=60)
    return fused_results[:top_k]
