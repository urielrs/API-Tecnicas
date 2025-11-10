from django.urls import path
from .views import AnalysisResultsAPIView

urlpatterns = [
    path('results/', AnalysisResultsAPIView.as_view(), name='analysis-results'),
]