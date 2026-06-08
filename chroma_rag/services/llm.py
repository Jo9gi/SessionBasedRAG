import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


def condense_question(chat_history: list, current_question: str) -> str:

    history_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question as a standalone question that can be understood without the history.
    Return ONLY the rewritten question, nothing else.
    Conversation History:
    {history_text}
    Follow-up Question: {current_question}
    Standalone Question:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def generate_response(query,retrieved_docs,chat_history=None):

    if not retrieved_docs:
        return "I could not find relevant information."

    context_parts = []
    for doc in retrieved_docs:
        file_name = doc.metadata.get("file_name", "Unknown Document")
        page = doc.metadata.get("page","Unknown")
        chapter = doc.metadata.get("chapter","General Content")
        context_parts.append(f"""
            DOCUMENT: {file_name}
            PAGE: {page}
            CHAPTER: {chapter}
            CONTENT:{doc.page_content}""")

    context = "\n\n".join(context_parts)
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
You are a document assistant.

Use ONLY the supplied context.

Instructions:

1. Answer using the retrieved document content.
2. Mention document names when useful.
3. Mention page numbers when useful.
4. If multiple documents contain relevant information, combine them.
5. If the answer is not present in the context, respond:

I could not find relevant information.

{f"Conversation History:{chr(10)}{history_text}" if history_text else ""}

CONTEXT:

{context}

QUESTION:

{query}

ANSWER:
"""
    response = llm.invoke(prompt)
    return response.content.strip()