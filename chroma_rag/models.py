from django.db import models
import uuid


# =====================================================
# SESSION MODEL
# =====================================================
# Each session represents one chat — tied to one uploaded document.
# When a user uploads a document, a new session is created.
# The session_id is a unique string used to identify the chat.
# The document_path stores where the uploaded file is saved on disk.
# The chroma_path stores where the vector DB for this session lives.
# =====================================================

class ChatSession(models.Model):
    session_id    = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    document_name = models.CharField(max_length=255)           # original file name shown in UI
    document_path = models.CharField(max_length=500)           # full path to the uploaded file on disk
    chroma_path   = models.CharField(max_length=500)           # path to this session's chroma vector DB
    created_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} — {self.document_name}"


# =====================================================
# CHAT MESSAGE MODEL
# =====================================================
# Each row is one message turn (user or assistant) inside a session.
# Linked to ChatSession via session_id.
# role is either "user" or "assistant".
# =====================================================

class ChatMessage(models.Model):
    session       = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role          = models.CharField(max_length=20)            # "user" or "assistant"
    content       = models.TextField()
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]                              # always return messages oldest-first

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"
