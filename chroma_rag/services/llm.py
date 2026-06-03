import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not found in context, say:
"I could not find relevant information."

CONTEXT:
{context}

QUESTION:
{query}
"""

    response = llm.invoke(prompt)
    return response.content
