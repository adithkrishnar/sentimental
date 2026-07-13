from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Mocking the load_model at import time to avoid file-not-found errors during setup
with patch("app.main.model_instance.load_model") as mock_load:
    from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("app.main.model_instance")
@patch("app.main.save_sentiment_prediction")
def test_predict_endpoint(mock_save, mock_model):
    mock_model.predict.return_value = {
        "text": "The movie was amazing!",
        "sentiment_prediction": "positive",
        "probabilities": {"positive": 0.85, "negative": 0.05, "neutral": 0.10},
        "latency_ms": 1.25
    }
    
    response = client.post("/predict", json={"text": "The movie was amazing!"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_prediction"] == "positive"
    assert data["probabilities"]["positive"] == 0.85
    assert "timestamp" in data
    mock_save.assert_called_once()

@patch("app.main.model_instance")
@patch("app.main.save_sentiment_prediction")
def test_batch_predict_endpoint(mock_save, mock_model):
    mock_model.predict_batch.return_value = [
        {"text": "Happy!", "sentiment_prediction": "positive", "probabilities": {"positive": 0.9}, "latency_ms": 0.8},
        {"text": "Sad.", "sentiment_prediction": "negative", "probabilities": {"negative": 0.9}, "latency_ms": 0.8}
    ]
    
    response = client.post("/batch_predict", json={"texts": ["Happy!", "Sad."]})
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["sentiment_prediction"] == "positive"
    assert data["predictions"][1]["sentiment_prediction"] == "negative"
    assert mock_save.call_count == 2

def test_model_metadata():
    response = client.get("/model")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert response.json()["model_name"] == "TF-IDF + LogisticRegression"
