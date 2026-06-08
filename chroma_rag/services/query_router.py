def detect_query_type(query):

    q = query.lower()

    if "chapter" in q:
        return "chapter"

    if "document count" in q:
        return "document_count"

    if "how many documents" in q:
        return "document_count"

    if "document names" in q:
        return "document_names"

    if "file names" in q:
        return "document_names"

    return "semantic"