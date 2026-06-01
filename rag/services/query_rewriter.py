import os
from langchain_openai import ChatOpenAI

rewriter_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# QUERY REWRITER
# =====================================================

def rewrite_query(query, domain="Machine Learning"):

    prompt = f"""You are a search query optimizer for a RAG system about {domain}.

Your job is to rewrite the user query into a clear, detailed, and retrieval-friendly query.

Rules:
- Expand abbreviations and vague terms
- Make the query self-contained and specific
- If the query is vague like "advantages?" or "its benefits?", assume it refers to {domain}
- Return ONLY the rewritten query, nothing else

Original Query: {query}

Rewritten Query:"""

    response = rewriter_llm.invoke(prompt)

    return response.content.strip()
