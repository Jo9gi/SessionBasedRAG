from chroma_rag.models import ChatSession, ChatMessage

def get_history(session_id: str) -> list:
    """
    Returns the chat history for a session as a list of {role, content} dicts.
    Returns empty list if the session is new or has no messages yet.
    """
    try:
        session = ChatSession.objects.get(session_id=session_id)
        messages = session.messages.values("role", "content")
        return list(messages)
    except ChatSession.DoesNotExist:
        return []


def add_to_history(session_id: str, user_message: str, assistant_message: str):
    """
    Saves the latest user question and assistant answer into the database.
    """
    session = ChatSession.objects.get(session_id=session_id)
    ChatMessage.objects.create(session=session, role="user",      content=user_message)
    ChatMessage.objects.create(session=session, role="assistant", content=assistant_message)


def clear_history(session_id: str):
    """
    Deletes all chat messages for a session (keeps the session itself).
    Called when user wants to reset the conversation without uploading a new document.
    """
    try:
        session = ChatSession.objects.get(session_id=session_id)
        session.messages.all().delete()
    except ChatSession.DoesNotExist:
        pass
