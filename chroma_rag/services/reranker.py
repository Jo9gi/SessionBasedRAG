from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(query, chunks, top_k=5):
    if not chunks:
        return chunks

    # build (query, chunk_text) pairs for the cross-encoder
    pairs = [[query, doc.page_content] for doc in chunks]

    # score each pair
    scores = reranker.predict(pairs)

    # attach scores and sort descending
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    print(f"\nRERANKER SCORES (top {top_k}):")
    for score, doc in scored[:top_k]:
        print(f"  score={score:.4f} | page={doc.metadata.get('page')}")

    return [doc for _, doc in scored[:top_k]]
