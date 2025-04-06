# DeepTruth - AI-Powered News Verification Platform

DeepTruth is a powerful backend service that leverages artificial intelligence to verify news articles and claims, helping users distinguish between factual and false information.

## 🌟 Features

- **False News Detection**: Analyzes news articles for potential misinformation
- **MongoDB Integration**: Efficient data storage and retrieval
- **RESTful API**: Well-structured endpoints for easy integration
- **Docker Support**: Containerized deployment for consistent environments

## 🚀 Tech Stack

- **Backend Framework**: Django 4.2.7
- **API Framework**: Django REST Framework 3.14.0
- **Database**: MongoDB with Djongo
- **AI/ML**: 
  - Google Generative AI
  - Transformers (DistilBERT)
  - Scikit-learn
- **Containerization**: Docker & Docker Compose
- **Other Tools**:
  - BeautifulSoup4 for web scraping
  - Python-dotenv for environment management
  - Requests for HTTP operations

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

DEBUG=True
```

3. Build and start the Docker containers:
```bash
docker-compose up --build
```

## 📝 API Endpoints

```

### 1. false news Verification
```bash
POST /api/verify-claim/

Request Body:
{
   'article_title': "the respective title"
}
```

## 🧪 Testing

Run the test suite:
```bash
docker-compose exec web python manage.py test
```

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
