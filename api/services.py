import requests
import os
import json
from typing import List, Dict, Any
from datetime import datetime
from django.utils import timezone
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,Trainer,TrainingArguments,DistilBertTokenizerFast
from datasets import Dataset
import torch

import logging

logger = logging.getLogger(__name__)

MODEL_DIR = r"E:\Deeptruth-Backend\api\distill_bert_model"

def retrain_distilbert_model(data):
    try:
        # Step 1: Prepare the dataset
        all_data = []

        for item in data["true"]:
            all_data.append({"text": item["title"], "label": 1})  # True → 1
        for item in data["false"]:
            all_data.append({"text": item["title"], "label": 0})  # False → 0

        if not all_data:
            logger.warning("No data provided for retraining.")
            return False

        # Step 2: Train/test split
        train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        # Step 3: Load tokenizer and model
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

        # Step 4: Tokenization function
        def tokenize_function(example):
            return tokenizer(example["text"], padding="max_length", truncation=True)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Step 5: Set format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Step 6: TrainingArguments
        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=os.path.join(MODEL_DIR, "logs"),
            logging_steps=10,
            load_best_model_at_end=True
        )

        # Step 7: Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Step 8: Retrain the model
        trainer.train()

        # Step 9: Save the retrained model
        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        logger.info("Retraining successful.")
        return True

    except Exception as e:
        logger.exception("Error during DistilBERT retraining")
        return False
class BraveNewsService:
    def __init__(self):
        self.api_key = os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is not set in environment variables")

    def get_news_articles(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }

        params = {'q': query}

        try:
            # Add delay to prevent rate limiting
            import time
            time.sleep(1)

            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10)
            response.raise_for_status()

            response_data = response.json()

            brave_news = []
            if 'web' in response_data and 'results' in response_data['web']:
                results = response_data['web']['results']

                for result in results[:limit]:
                    try:
                        article = {
                            'title': result.get('title', ''),
                            'link': result.get('url', ''),
                            'snippet': result.get('description', ''),
                            'source': result.get('site_name', 'Unknown Source'),
                            'time_published': result.get('published_date', 'Unknown Date'),
                            'retrieved_at': datetime.now().isoformat()
                        }
                        brave_news.append(article)
                    except Exception as e:
                        print(f"Error parsing individual result: {e}")
                        continue

            # If we couldn't find any valid results, return a default response
            if not brave_news:
                return [{
                    'title': 'No results found',
                    'link': '',
                    'snippet': 'Could not find relevant news articles',
                    'source': 'System',
                    'time_published': datetime.now().isoformat(),
                    'retrieved_at': datetime.now().isoformat()
                }]

            return brave_news

        except Exception as e:
            print(f"Error fetching news from Brave API: {e}")
            # Return a default response in case of error
            return [{
                'title': 'Error fetching news',
                'link': '',
                'snippet': f'An error occurred while fetching news: {str(e)}',
                'source': 'System',
                'time_published': datetime.now().isoformat(),
                'retrieved_at': datetime.now().isoformat()
            }]

class DistilBERTService:
    def __init__(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'distil_bert_model')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2  # Binary classification (true/false)
            )
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            print(f"Error initializing DistilBERT model: {str(e)}")
            raise ValueError(f"Failed to initialize DistilBERT model: {str(e)}")

    def analyze_claim(self, title: str) -> float:
        """
        Analyze the claim using DistilBERT model
        Returns confidence score between 0 and 1
        """
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                title,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence_score = probabilities[0][1].item()  # Assuming 1 is true class
            return confidence_score
        except Exception as e:
            print(f"DistilBERT Error: {str(e)}")
            return 0.5  # Return neutral score on error
class GeminiService:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"

    def clean_json_text(self, text: str) -> str:
        """Clean the text to make it JSON-compatible"""
        # Remove backticks and other special characters
        text = text.replace('`', '')
        # Remove any control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        # Ensure proper JSON formatting
        text = text.replace('\n', ' ').replace('\r', '')
        # Remove any extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text"""
        # Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in text")
        return text[start:end]

    def analyze_claim(self, title: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Format the context from news articles
            context = "\n\n".join([
                f"Source {i+1}:\nTitle: {article.get('title', '')}\nDescription: {article.get('snippet', '')}\nURL: {article.get('link', '')}\n"
                for i, article in enumerate(news_articles)
            ])

            # Create the prompt
            prompt = f"""Analyze this claim: "{title}"

Context from news sources:
{context}

Provide a detailed fact-check analysis in JSON format with the following fields:
- veracity (boolean): true if claim is verified, false if misleading
- confidence_score (float between 0-1): how confident the analysis is
- explanation (string, min 250 words): detailed analysis referencing sources
- category (string): type of claim
- key_findings (list): main points from analysis
- impact_level (string): one of ["VERIFIED", "MISLEADING", "PARTIAL"]
- sources (list): URLs of relevant sources

Requirements:
1. Reference each source by name/number in explanation
2. Connect evidence across sources
3. Include specific quotes/data points
4. Explain reasoning for confidence score
5. Minimum 250-word explanation
6. Clear true/false determination
7. List key findings with citations"""

            # Prepare the request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }

            # Make the API request
            headers = {
                'Content-Type': 'application/json'
            }

            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    response_data = response.json()

                    # Extract the generated text from the response
                    if 'candidates' in response_data and response_data['candidates']:
                        generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        raise ValueError("No response content found in Gemini API response")

                    # Clean and parse the response
                    try:
                        # First attempt: try to parse the cleaned text directly
                        cleaned_text = self.clean_json_text(generated_text)
                        result = json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        try:
                            # Second attempt: try to extract and parse JSON object
                            json_text = self.extract_json_from_text(cleaned_text)
                            result = json.loads(json_text)
                        except json.JSONDecodeError as e:
                            print(f"JSON Parsing Error: {str(e)}")
                            print(f"Cleaned Text: {cleaned_text}")
                            raise ValueError("Could not parse JSON from response")

                    # Validate required fields
                    required_fields = ['veracity', 'confidence_score', 'explanation', 'category', 
                                    'key_findings', 'impact_level', 'sources']
                    missing_fields = [field for field in required_fields if field not in result]
                    if missing_fields:
                        raise ValueError(f"Missing required fields in response: {missing_fields}")

                    # Ensure confidence score is between 0 and 1
                    result['confidence_score'] = max(0.0, min(1.0, float(result['confidence_score'])))

                    return result
                    
                except requests.exceptions.RequestException as e:
                    if response.status_code == 503 and attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
                        continue
                    print(f"API Request Error: {str(e)}")
                    return self._error_response(f"API request failed: {str(e)}")
                
                except Exception as e:
                    print(f"Gemini Error: {str(e)}")
                    return self._error_response(f"Error processing request: {str(e)}")
            
            return self._error_response("Maximum retries exceeded")

        except Exception as e:
            print(f"Gemini Service Error: {str(e)}")
            return self._error_response(f"Error in analyze_claim: {str(e)}")
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "veracity": False,
            "confidence_score": 0.0,
            "explanation": error_message,
            "category": "Error",
            "key_findings": [],
            "impact_level": "MISLEADING",
            "sources": []
        }
def retrain_distilbert_model(data):
    try:
        # Simulate retraining
        print("Retraining on:", data)
        # TODO: Insert your real retraining logic here (HuggingFace, transformers, etc.)
        return True  # Return False to simulate failure
    except Exception as e:
        print("Error during retraining:", e)
        return False

class CombinedAnalysisService:
    def __init__(self):
        self.brave_service = BraveNewsService()
        self.gemini_service = GeminiService()

    def analyze_claim(self, article_title: str) -> Dict[str, Any]:
        """
        Analyze a claim using Brave search and Gemini analysis
        """
        try:
            # Get news articles from Brave Search
            brave_news = self.brave_service.get_news_articles(article_title)
            
            # Get Gemini analysis
            gemini_result = self.gemini_service.analyze_claim(article_title, brave_news)
            
            distilbert_score = self.distilbert_service.analyze_claim(article_title)
            
            # Process sources from Gemini and Brave
            gemini_sources = gemini_result.get('sources', [])
            brave_sources = [article.get('link', '') for article in brave_news if article.get('link')]
            
            # Combine and deduplicate sources
            all_sources = list(set(gemini_sources + brave_sources))
            
            # Create result dictionary
            result = {
                'veracity': gemini_result['veracity'],
                'confidence_score': (gemini_result['confidence_score'](0.6) + distilbert_score(0.4)) / 2,
                'explanation': gemini_result['explanation'],
                'category': gemini_result['category'],
                'key_findings': gemini_result['key_findings'],
                'impact_level': gemini_result['impact_level'],
                'sources': all_sources,  # Use combined and deduplicated sources
                'created_at': timezone.now(),
                'updated_at': timezone.now()
            }
            
            return result

        except Exception as e:
            print(f"Combined Analysis Error: {str(e)}")
            return {
                'veracity': False,
                'confidence_score': 0.5,
                'explanation': f"Error during analysis: {str(e)}",
                'category': 'ERROR',
                'key_findings': [],
                'impact_level': 'PARTIAL',
                'sources': [],
                'created_at': timezone.now(),
                'updated_at': timezone.now()
            }
