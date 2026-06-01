def retrieve_chunks(query, vector_store, top_k=5, filter=None):

    search_kwargs = {"k": top_k}

    if filter:
        search_kwargs["expr"] = filter

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    results = retriever.invoke(query)

    return results