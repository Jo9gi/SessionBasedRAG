from django.urls import path
from .views import (
    RAGUploadAPIView,
    RAGQueryAPIView,
    RAGSessionHistoryAPIView,
    RAGClearSessionAPIView,
    RAGUIView,
)

urlpatterns = [
    path('',                            RAGUIView.as_view(),             name='rag_ui'),

    # Upload a document → returns session_id
    path('upload/',                     RAGUploadAPIView.as_view(),      name='rag_upload'),

    # Send a query with session_id → returns answer
    path('query/',                      RAGQueryAPIView.as_view(),       name='rag_query'),

    # Get full chat history for a session → used when opening an old chat
    path('session/<str:session_id>/',   RAGSessionHistoryAPIView.as_view(), name='rag_session_history'),

    # Clear chat messages for a session (keeps document + vectors intact)
    path('clear-session/',              RAGClearSessionAPIView.as_view(), name='rag_clear_session'),
]
