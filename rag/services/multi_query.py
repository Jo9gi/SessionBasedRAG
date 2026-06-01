import os
from langchain_openai import ChatOpenAI


rewriter_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# GENERATE MULTIPLE QUERY VARIANTS
# =====================================================

def generate_queries(query, domain="Machine Learning", n=4):

    prompt = f"""You are a search query generator for a RAG system about {domain}.

Generate {n} different versions of the query below.
Each version should approach the topic from a different angle.

Rules:
- Each query must be on a new line
- No numbering, no bullets, no extra text
- Keep each query concise and specific

Original Query: {query}

Queries:"""

    response = rewriter_llm.invoke(prompt)

    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    # always include the original
    if query not in queries:
        queries.insert(0, query)

    return queries[:n]


# =====================================================
# RECIPROCAL RANK FUSION
# =====================================================

def reciprocal_rank_fusion(results_per_query, k=60):

    scores = {}
    doc_map = {}

    for results in results_per_query:

        for rank, doc in enumerate(results):

            doc_id = doc.page_content

            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = doc

            scores[doc_id] += 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[doc_id] for doc_id, _ in ranked]


# =====================================================
# MULTI QUERY RETRIEVAL
# =====================================================

def multi_query_retrieve(query, vector_store, chunks, top_k=5, filter=None, domain="Machine Learning"):

    queries = generate_queries(query, domain=domain)

    print(f"\nGENERATED QUERIES:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    from .hybrid_search import hybrid_search

    results_per_query = []

    for q in queries:
        results = hybrid_search(query=q, vector_store=vector_store, chunks=chunks, top_k=top_k, filter=filter)
        results_per_query.append(results)

    fused = reciprocal_rank_fusion(results_per_query)

    return fused[:top_k]
