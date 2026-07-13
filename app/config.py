import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sentiment_pipeline.joblib")
    METRICS_PATH: str = os.getenv("METRICS_PATH", "models/metrics.json")
    
    # AWS configuration for S3 storage
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "sentiment-predictions-storage")
    
    # Local fallback storage path
    LOCAL_STORAGE_DIR: str = os.getenv("LOCAL_STORAGE_DIR", "data/predictions")

settings = Settings()
