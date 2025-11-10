# data_api_project/views.py
from django.shortcuts import render

def index_view(request):
    """Renderiza el archivo index.html."""
    return render(request, 'index.html', {})