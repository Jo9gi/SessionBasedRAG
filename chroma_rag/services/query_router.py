def detect_query_type(query):

    q = query.lower()

    # Chapter / TOC queries
    if (
        "chapter" in q
        or "chapters" in q
        or "table of contents" in q
        or "toc" in q
    ):
        return "chapter_names"

    # Document count queries
    if (
        "how many documents" in q
        or "document count" in q
        or "number of documents" in q
    ):
        return "document_count"

    # Document name queries
    if (
        "document names" in q
        or "file names" in q
        or "list documents" in q
        or "which documents" in q
    ):
        return "document_names"

    return "semantic"