# 🚀 Deployment Guide

This guide covers multiple deployment options for the Romance Language Classifier.

## 📋 Prerequisites

- Python 3.8+
- Git
- Docker (for containerized deployment)
- A cloud platform account (AWS, GCP, Azure, etc.)

## 🎯 Deployment Options

### 1. Streamlit Cloud (Recommended for Demo)

**Pros:** Free, easy setup, perfect for portfolios
**Cons:** Limited resources, no custom domain

#### Setup Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add enhanced romance classifier"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file path: `src/streamlit.py`
   - Deploy!

3. **Update README**
   - Update the live demo link in your README
   - Add deployment badges

### 2. Docker Deployment

**Pros:** Consistent environment, scalable, professional
**Cons:** Requires Docker knowledge

#### Local Docker:
```bash
# Build image
docker build -t romance-classifier .

# Run API
docker run -p 8000:8000 romance-classifier

# Run with docker-compose
docker-compose up -d
```

#### Cloud Deployment (AWS/GCP/Azure):
```bash
# Build and push to container registry
docker tag romance-classifier your-registry/romance-classifier:latest
docker push your-registry/romance-classifier:latest

# Deploy to cloud platform
# (Platform-specific commands)
```

### 3. Heroku Deployment

**Pros:** Easy deployment, good free tier
**Cons:** Limited resources, may need paid plan for ML models

#### Setup:
1. Create `Procfile`:
   ```
   web: uvicorn src.api:app --host=0.0.0.0 --port=$PORT
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### 4. AWS/GCP/Azure Deployment

**Pros:** Scalable, professional, full control
**Cons:** More complex, costs money

#### AWS Example (EC2):
```bash
# Launch EC2 instance
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# Pull and run
docker pull your-registry/romance-classifier
docker run -d -p 80:8000 romance-classifier
```

## 🔧 Environment Configuration

### Environment Variables
Create `.env` file:
```env
MODEL_PATH=models/best_model.pth
VOCAB_PATH=models/vocab.json
LOG_LEVEL=INFO
```

### Production Settings
Update `src/api.py`:
```python
# Add production settings
if os.getenv("ENVIRONMENT") == "production":
    app.debug = False
    # Add security headers
    # Add rate limiting
    # Add logging
```

## 📊 Monitoring & Analytics

### 1. Add Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add to API endpoints
logger.info(f"Classification request: {request.text[:50]}...")
```

### 2. Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "memory_usage": psutil.virtual_memory().percent
    }
```

### 3. Metrics Collection
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

classification_requests = Counter('classification_requests_total', 'Total classification requests')
classification_duration = Histogram('classification_duration_seconds', 'Classification duration')
```

## 🔒 Security Considerations

### 1. Input Validation
```python
from pydantic import validator

class ClassificationRequest(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > 1000:
            raise ValueError('Text too long')
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
```

### 2. Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/classify")
@limiter.limit("10/minute")
async def classify_text(request: ClassificationRequest):
    # Your code here
```

### 3. CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📈 Performance Optimization

### 1. Model Optimization
```python
# Quantize model for faster inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Use TorchScript for faster inference
traced_model = torch.jit.trace(model, example_input)
traced_model.save("optimized_model.pt")
```

### 2. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classification(text):
    return classify_text(text)
```

### 3. Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/classify_batch")
async def classify_batch(texts: List[str]):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, classify_single, text) for text in texts]
    return await asyncio.gather(*tasks)
```

## 🧪 Testing

### 1. Unit Tests
```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_classify_text():
    response = client.post("/classify", json={"text": "Hello world"})
    assert response.status_code == 200
    assert "prediction" in response.json()
```

### 2. Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

## 📝 Deployment Checklist

- [ ] Code is tested and working
- [ ] Environment variables configured
- [ ] Security measures implemented
- [ ] Monitoring setup
- [ ] Documentation updated
- [ ] Performance optimized
- [ ] Backup strategy in place
- [ ] SSL certificate configured
- [ ] Domain configured
- [ ] Analytics tracking setup

## 🚨 Troubleshooting

### Common Issues:

1. **Model Loading Errors**
   - Check file paths
   - Verify model file exists
   - Check PyTorch version compatibility

2. **Memory Issues**
   - Reduce batch size
   - Use model quantization
   - Increase server memory

3. **Performance Issues**
   - Enable caching
   - Optimize model
   - Use async processing

4. **Deployment Failures**
   - Check logs
   - Verify dependencies
   - Test locally first

## 📞 Support

For deployment issues:
1. Check the logs
2. Test locally
3. Review this guide
4. Open an issue on GitHub

---

**Remember:** Always test your deployment in a staging environment first!




