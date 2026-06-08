from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def header_recursive_chunking(documents, chunk_size=1500, chunk_overlap=200):
    """Split documents using headers first, then recursively split each section.

    Steps:
    1. **Header split** – respects markdown/HTML headings (e.g., "#", "##") so that chunks align with natural sections such as chapters.
    2. **Recursive split** – further breaks any large sections into manageable 1500‑character pieces with overlap to preserve context.
    """
    # First split by headings (supports common markdown header levels)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "chapter"), ("##", "section"), ("###", "subsection")],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    header_chunks = header_splitter.split_documents(documents)

    # Now recursively split each header chunk to ensure no piece exceeds the desired size
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    final_chunks = []
    for chunk in header_chunks:
        final_chunks.extend(recursive_splitter.split_documents([chunk]))
    return final_chunks
    # Compatibility alias
    def recursive_chunking(documents, chunk_size=1500, chunk_overlap=200):
        """Legacy wrapper that forwards to header_recursive_chunking."""
        return header_recursive_chunking(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
