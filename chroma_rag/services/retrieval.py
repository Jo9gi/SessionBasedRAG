import re

# =====================================================
# CHROMA RETRIEVAL SERVICE
# =====================================================

def retrieve_chunks(query, vector_store, top_k=5, filter=None):
    search_kwargs = {"k": top_k}

    if filter:
        # Translate Milvus string filter format (e.g., 'metadata["page"] == 5')
        # into Chroma's dict filter format (e.g., {"page": 5})
        if "page" in filter:
            try:
                match = re.search(r'\d+', filter)
                if match:
                    page_num = int(match.group())
                    search_kwargs["filter"] = {"page": page_num}
            except Exception as e:
                print(f"Error parsing metadata filter for Chroma: {e}")
        else:
            # Fallback to direct assignment
            search_kwargs["filter"] = filter

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    results = retriever.invoke(query)
    return results
