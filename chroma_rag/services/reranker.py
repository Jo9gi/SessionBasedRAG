from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =====================================================
# SCORE THRESHOLD
# =====================================================
# The ms-marco cross-encoder scores range roughly from -12 to +12.
#   score > 0   → chunk is likely relevant to the query
#   score 0 to -3 → weakly relevant (borderline)
#   score < -3  → chunk is essentially irrelevant noise
#
# Without a threshold, the LLM receives junk chunks and either:
#   (a) hallucinates an answer from loose word associations, OR
#   (b) gets confused by irrelevant context and refuses to answer
#
# With MIN_SCORE = -3.0:
#   - Very negative scores (-8, -10, etc.) are filtered out
#   - If ALL chunks are filtered, rerank_chunks returns [] 
#     → views.py / test_rag.py catches empty list → "no relevant info"
# =====================================================

MIN_SCORE = -3.0


def rerank_chunks(query, chunks, top_k=6):
    if not chunks:
        return []

    # Score every (query, chunk) pair with the cross-encoder
    pairs = [[query, doc.page_content] for doc in chunks]
    scores = reranker.predict(pairs)

    # Sort descending by score
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    # Print all scores so you can see the full picture during debugging
    print(f"\nRERANKER SCORES (top {top_k}):")
    for score, doc in scored[:top_k]:
        label = "GOOD" if score > 0 else ("WEAK" if score > MIN_SCORE else "NOISE")
        print(f"  score={score:.4f} [{label}] | page={doc.metadata.get('page')}")

    # Filter: only keep chunks that score above MIN_SCORE threshold
    # This prevents irrelevant chunks from reaching the LLM
    filtered = [(score, doc) for score, doc in scored[:top_k] if score >= MIN_SCORE]

    if not filtered:
        print(f"\n  [WARNING] All {top_k} chunks scored below threshold ({MIN_SCORE}).")
        print("  [WARNING] Returning empty — LLM will respond 'I could not find relevant information.'")
        return []

    kept = len(filtered)
    dropped = top_k - kept if top_k <= len(scored) else len(scored) - kept
    if dropped > 0:
        print(f"\n  [INFO] Kept {kept} relevant chunk(s), dropped {dropped} noise chunk(s) (score < {MIN_SCORE})")

    return [doc for _, doc in filtered]
