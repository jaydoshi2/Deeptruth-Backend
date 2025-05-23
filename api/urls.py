from django.urls import path
from .views import RetrainModelAPIView, VerifyClaimAPIView, FalseNewsListView

urlpatterns = [
    path('verify-claim/', VerifyClaimAPIView.as_view(), name='verify-claim'),
    path('false-news/', FalseNewsListView.as_view(), name='false-news-list'),
    path('retrain-model/', RetrainModelAPIView.as_view(), name='retrain-model'),
] 