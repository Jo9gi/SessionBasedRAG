from django.db import models
import uuid

class ChatSession(models.Model):
    session_id    = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    session_name  = models.CharField(max_length=255, blank=True, null=True)
    chroma_path   = models.CharField(max_length=500)           
    created_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session_id} — {self.session_name}"

class SessionDocument(models.Model):
    session       = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="documents")
    document_name = models.CharField(max_length=255)
    document_path = models.CharField(max_length=500)
    created_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session.session_id} -> {self.document_name}"

class ChatMessage(models.Model):
    session       = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role          = models.CharField(max_length=20)            
    content       = models.TextField()
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]                              

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"
