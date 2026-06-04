from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('', RedirectView.as_view(url='/api/chroma/', permanent=False)),
    path('admin/', admin.site.urls),
    path('api/chroma/', include('chroma_rag.urls')),
]
