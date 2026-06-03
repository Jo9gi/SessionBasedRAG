import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

rewriter_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


def rewrite_query(query, file_context=None):
    """
    Rewrites the user query to optimize it for vector and keyword search.
    Adapts dynamically to the domain of the active uploaded document.
    """
    context_str = ""
    if file_context:
        context_str = f" The search is running against the active document with filename: '{file_context}' (use this name to infer the subject matter/domain, but do NOT include the filename, extension, or the word 'file/document' in the output)."

    prompt = f"""You are an advanced search query optimizer for a Retrieval-Augmented Generation (RAG) system.{context_str}

Your job is to rewrite the user query into a clear, detailed, and search-friendly query.

Rules:
- Identify the target domain/subject from the query (e.g., medical, finance, programming, history, laws).
- Expand abbreviations and technical terms relevant to that domain.
- Keep the query self-contained, specific, and optimized for search.
- Do NOT include the filename, file extension, or the word "file/document" in the rewritten query. Focus purely on the content (e.g., if searching for a figure, refer to it directly as "Figure X-Y").
- Return ONLY the rewritten query text, nothing else.

Original Query: {query}

Rewritten Query:"""

    response = rewriter_llm.invoke(prompt)
    return response.content.strip()
