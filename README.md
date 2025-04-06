# DeepTruth - AI-Powered News Verification Platform

DeepTruth is a powerful backend service that leverages artificial intelligence to verify news articles and claims, helping users distinguish between factual and false information.

## 🌟 Features

- **False News Detection**: Analyzes news articles for potential misinformation
- **Claim Verification**: Verifies specific claims using AI and fact-checking
- **MongoDB Integration**: Efficient data storage and retrieval
- **RESTful API**: Well-structured endpoints for easy integration
- **Docker Support**: Containerized deployment for consistent environments
- **AI-Powered Analysis**: Uses Google Generative AI and Transformers for content analysis

## 🚀 Tech Stack

- **Backend Framework**: Django
- **API Framework**: Django REST Framework 3.12.4
- **Database**: MongoDB with Djongo 1.3.6
- **AI/ML**: 
  - Google Generative AI 0.1.0rc1
  - Transformers 4.44.0
  - Scikit-learn 1.3.2
  - PyTorch 2.4.0
- **Containerization**: Docker & Docker Compose
- **Other Tools**:
  - BeautifulSoup4 4.12.3 for web scraping
  - Python-dotenv 1.0.1 for environment management
  - Requests 2.32.3 for HTTP operations
  - Gunicorn 20.1.0 for production server

## 🛠️ Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher (if running locally)
- MongoDB Atlas account (for cloud database)
- sckit-learn libarary

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deeptruth-backend.git
cd deeptruth-backend
```

2. Create a `.env` file in the root directory with the following variables:
```env
# Google Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key
BRAVE_API_KEY=your_brave_api_key

# MongoDB Configuration
MONGODB_NAME=deeptruth_db
MONGODB_URI=your_mongodb_uri
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password

# Django Configuration
DEBUG=True
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=localhost,127.0.0.1
```

3. Build and start the Docker containers:
```bash
docker-compose up --build
```

## 📊 Data Models

### FalseNews
- `article_title`: Title of the news article
- `veracity`: Boolean indicating if the news is false
- `confidence_score`: AI model's confidence in the verification
- `explanation`: Detailed explanation of the verification
- `category`: News category
- `key_findings`: JSON field containing key findings
- `impact_level`: Verification status (VERIFIED/MISLEADING/PARTIAL)
- `sources`: List of source URLs
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### TrueNews
- Similar structure to FalseNews but with `veracity` defaulting to True
- Additional fields for storing verified true information

## 📝 API Endpoints

### 1. False News Detection
```bash
POST /api/false-news/

Request Body:
{
    "url": "https://example.com/news-article",
    "title": "Example News Title",
    "content": "Article content...",
    "source": "News Source",
    "published_date": "2024-03-21T12:00:00Z"
}
```

### 2. Claim Verification
```bash
POST /api/true-news/

Request Body:
{
    "claim": "The claim to verify",
    "context": "Additional context",
    "source": "Source of the claim",
    "language": "en"
}
```

## 🧪 Testing

Run the test suite:
```bash
docker-compose exec web python manage.py test
```

## 📚 Documentation

For detailed API documentation, visit:
```
http://localhost:8000/api/docs/
```

## 🔒 Security

- All API endpoints are protected with authentication
- Environment variables for sensitive data
- Secure MongoDB connection with authentication

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Google Gemini AI for natural language processing capabilities
- MongoDB Atlas for database hosting
- Django and DRF communities for excellent frameworks
