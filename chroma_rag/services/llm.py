import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# CONDENSE QUESTION
# =====================================================
# When a user asks a follow-up question like "What about section 2?",
# the system needs to understand what "section 2" refers to from history.
# This function rewrites the follow-up into a fully self-contained question
# so the retrieval step can find the right chunks.
#
# Example:
#   History : Q: "What is the refund policy?" A: "Refunds within 30 days..."
#   Follow-up: "What if I paid by card?"
#   Condensed: "What is the refund policy for card payments?"
# =====================================================

def condense_question(chat_history: list, current_question: str) -> str:
    """
    Rewrites the current question into a standalone question using chat history.
    Only called when chat history exists (not on the first query).
    """
    # Format the history as a readable conversation string
    history_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Given the following conversation history and a follow-up question, \
rewrite the follow-up question as a standalone question that can be understood without the history.
Return ONLY the rewritten question, nothing else.

Conversation History:
{history_text}
Follow-up Question: {current_question}
Standalone Question:"""

    response = llm.invoke(prompt)
    return response.content.strip()


# =====================================================
# GENERATE RESPONSE
# =====================================================
# Answers the user query strictly from the retrieved document chunks.
# Chat history is included so the LLM can give a coherent, context-aware answer.
# =====================================================

def generate_response(query, retrieved_docs, chat_history=None):
    context = "\n\n".join([doc.page_content[:800] for doc in retrieved_docs])

    # Build conversation history string to include in the prompt (if any)
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful AI assistant.

Answer ONLY from the provided context.
If the answer is not found in the context, say: "I could not find relevant information."

{f'Conversation so far:{chr(10)}{history_text}' if history_text else ''}
CONTEXT:
{context}

QUESTION:
{query}
"""

    response = llm.invoke(prompt)
    return response.content
