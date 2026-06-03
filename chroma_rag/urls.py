from django.urls import path
from .views import RAGQueryAPIView, RAGIndexAPIView, RAGUploadAPIView

urlpatterns = [
    path('query/', RAGQueryAPIView.as_view(), name='rag_query'),
    path('index/', RAGIndexAPIView.as_view(), name='rag_index'),
    path('upload/', RAGUploadAPIView.as_view(), name='rag_upload'),
]
