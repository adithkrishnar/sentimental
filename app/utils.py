import os
import json
import uuid
from datetime import datetime, timezone
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from app.config import settings

def save_sentiment_prediction(prediction_payload: dict) -> str:
    """
    Saves a review sentiment prediction record to S3 if credentials are provided,
    otherwise falls back to saving to a local JSON file in the data/predictions folder.
    Returns the path or S3 key where the payload was saved.
    """
    filename = f"prediction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    
    # Check if AWS credentials and bucket are provided
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            s3_key = f"sentiment-predictions/{filename}"
            s3.put_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=s3_key,
                Body=json.dumps(prediction_payload, indent=2),
                ContentType="application/json"
            )
            print(f"Successfully uploaded prediction to S3: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
            return f"s3://{settings.S3_BUCKET_NAME}/{s3_key}"
        except (NoCredentialsError, ClientError) as e:
            print(f"S3 upload failed: {str(e)}. Falling back to local storage.")
    
    # Local fallback
    os.makedirs(settings.LOCAL_STORAGE_DIR, exist_ok=True)
    local_path = os.path.join(settings.LOCAL_STORAGE_DIR, filename)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(prediction_payload, f, indent=2)
    print(f"Successfully saved prediction locally to: {local_path}")
    return local_path
