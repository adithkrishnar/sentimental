import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from app.model import model_instance
from app.schemas import (
    ReviewRequest,
    PredictionResponse,
    BatchReviewRequest,
    BatchPredictionResponse,
    ModelMetadataResponse
)
from app.utils import save_sentiment_prediction
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Loading Sentiment classification model on startup...")
    try:
        model_instance.load_model()
        print("Sentiment pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model during startup: {str(e)}")
        print("Note: If running for the first time, you must run train.py to generate artifacts.")
    yield
    print("Shutting down API...")

app = FastAPI(
    title="Sentiment Analysis & Review Intelligence API",
    description="MLOps service serving a TF-IDF + Logistic Regression model for sentiment classification.",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Health Check"])
def health_check():
    """
    Health check endpoint to verify API and model status.
    """
    storage_type = "S3" if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY else "Local File System"
    model_loaded = model_instance.pipeline is not None
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_loaded,
        "storage_mode": storage_type
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(review: ReviewRequest, background_tasks: BackgroundTasks):
    """
    Predict sentiment class (positive/negative/neutral) and probabilities for a single text.
    The prediction results are saved asynchronously in the background.
    """
    try:
        result = model_instance.predict(review.text)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Save to S3 / Local fallback
        background_tasks.add_task(save_sentiment_prediction, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(batch: BatchReviewRequest, background_tasks: BackgroundTasks):
    """
    Predict sentiment classes and probabilities for a batch of reviews.
    All prediction results are saved asynchronously in the background.
    """
    try:
        start_time = time.time()
        
        predictions = model_instance.predict_batch(batch.texts)
        total_latency_ms = (time.time() - start_time) * 1000
        timestamp = datetime.now(timezone.utc).isoformat()
        
        formatted_predictions = []
        for pred in predictions:
            pred["timestamp"] = timestamp
            formatted_predictions.append(pred)
            background_tasks.add_task(save_sentiment_prediction, pred)
            
        return {
            "predictions": formatted_predictions,
            "total_reviews": len(batch.texts),
            "latency_ms": round(total_latency_ms, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

@app.get("/model", response_model=ModelMetadataResponse, tags=["Metadata"])
def get_model_metadata():
    """
    Returns training metrics and loading details about the model.
    """
    status = "loaded" if model_instance.pipeline is not None else "not_loaded"
    return {
        "model_name": "TF-IDF + LogisticRegression",
        "status": status,
        "metrics": model_instance.metrics
    }
