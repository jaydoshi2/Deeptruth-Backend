import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import ListAPIView
from django.db.models import Q
from .serializers import FalseNewsSerializer, TrueNewsSerializer
from .models import FalseNews, TrueNews
# from .services import CombinedAnalysisService, BraveNewsService, ModelTrainingService
from .services import BraveNewsService, GeminiService, retrain_distilbert_model
from datetime import datetime
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class RetrainModelAPIView(APIView):
    def post(self, request):
        try:
            # Fetch all True and False news entries
            true_news_qs = TrueNews.objects.all()
            false_news_qs = FalseNews.objects.all()

            # Serialize the data
            true_data = TrueNewsSerializer(true_news_qs, many=True).data
            false_data = FalseNewsSerializer(false_news_qs, many=True).data

            # Extract titles and confidence scores only
            train_data = {
                "true": [
                    {"title": news["article_title"], "confidence": news["confidence_score"]}
                    for news in true_data
                ],
                "false": [
                    {"title": news["article_title"], "confidence": news["confidence_score"]}
                    for news in false_data
                ]
            }

            # === Call your retraining function here ===
            # Replace the below line with actual model retraining logic
            retraining_successful = retrain_distilbert_model(train_data)

            if retraining_successful:
                TrueNews.objects.all().delete()
                FalseNews.objects.all().delete()
                return Response({"message": "Retraining successful. Collections cleared."}, status=status.HTTP_200_OK)
            else:
                logger.error("Retraining failed. Collections were not cleared.")
                return Response({"error": "Retraining failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.exception("Retraining process encountered an error.")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class VerifyClaimAPIView(APIView):
    def post(self, request):
        article_title = request.data.get('article_title')
        if not article_title:
            return Response(
                {'error': 'Article title is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Initialize services
            brave_service = BraveNewsService()
            gemini_service = GeminiService()

            # Get news articles from Brave
            news_articles = brave_service.get_news_articles(article_title)

            # Analyze claim using Gemini
            analysis_result = gemini_service.analyze_claim(article_title, news_articles)

            # Determine which model to save in based on veracity
            if analysis_result.get('veracity', False):
                true_news = TrueNews.objects.create(
                    article_title=article_title,
                    veracity=True,
                    confidence_score=analysis_result.get('confidence_score', 0.0),
                    explanation=analysis_result.get('explanation', ''),
                    category=analysis_result.get('category', 'Unknown'),
                    key_findings=analysis_result.get('key_findings', []),
                    impact_level=analysis_result.get('impact_level', 'VERIFIED'),
                    sources=analysis_result.get('sources', [])
                )
                serializer = TrueNewsSerializer(true_news)
            else:
                false_news = FalseNews.objects.create(
                    article_title=article_title,
                    veracity=False,
                    confidence_score=analysis_result.get('confidence_score', 0.0),
                    explanation=analysis_result.get('explanation', ''),
                    category=analysis_result.get('category', 'Unknown'),
                    key_findings=analysis_result.get('key_findings', []),
                    impact_level=analysis_result.get('impact_level', 'PARTIAL'),
                    sources=analysis_result.get('sources', [])
                )
                serializer = FalseNewsSerializer(false_news)

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FalseNewsListView(ListAPIView):
    serializer_class = FalseNewsSerializer
    
    def get_queryset(self):
        queryset = FalseNews.objects.all()
        
        # Filter by category
        category = self.request.query_params.get('category', None)
        if category:
            queryset = queryset.filter(category=category)
        
        # Filter by impact level
        impact = self.request.query_params.get('impact', None)
        if impact:
            queryset = queryset.filter(impact_level=impact)
        
        # Filter by fact check status
        status = self.request.query_params.get('status', None)
        if status:
            queryset = queryset.filter(fact_check_status=status)
        
        # Search by title or explanation
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(article_title__icontains=search) |
                Q(explanation__icontains=search) |
                Q(tags__contains=[search.lower()])
            )
        
        return queryset.order_by('-verification_date') 