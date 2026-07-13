from pydantic import BaseModel, Field
from typing import List, Dict

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to evaluate for sentiment")

class BatchReviewRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of text contents to evaluate for sentiment")

class PredictionResponse(BaseModel):
    text: str
    sentiment_prediction: str = Field(..., description="Predicted sentiment class (positive, negative, neutral)")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all classes")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_reviews: int
    latency_ms: float

class ModelMetadataResponse(BaseModel):
    model_name: str
    status: str
    metrics: dict
