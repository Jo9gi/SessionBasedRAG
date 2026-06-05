from django.urls import path
from .views import (
    RAGUploadAPIView,
    RAGQueryAPIView,
    RAGSessionHistoryAPIView,
    RAGClearSessionAPIView,
    RAGDeleteSessionAPIView,
    RAGUIView,
)

urlpatterns = [
    path('',RAGUIView.as_view(),name='rag_ui'),

    path('upload/',RAGUploadAPIView.as_view(),name='rag_upload'),

    path('query/',RAGQueryAPIView.as_view(),name='rag_query'),

    path('session/<str:session_id>/',RAGSessionHistoryAPIView.as_view(), name='rag_session_history'),

    path('clear-session/',RAGClearSessionAPIView.as_view(), name='rag_clear_session'),

    path('delete-session/',RAGDeleteSessionAPIView.as_view(), name='rag_delete_session'),
]
