

def retrieve_chunks(query, vector_store, top_k=15):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(query)
    return results
