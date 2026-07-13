import os
import json
import time
from typing import List, Dict, Any
import joblib
from app.config import settings

class SentimentModel:
    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.metrics_path = settings.METRICS_PATH
        
        self.pipeline = None
        self.metrics = {}

    def load_model(self):
        """
        Loads the TF-IDF + Classifier pipeline.
        This runs on FastAPI startup.
        """
        if self.pipeline is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model pipeline file not found at {self.model_path}. "
                    "Make sure to run src/train.py first."
                )
            
            self.pipeline = joblib.load(self.model_path)
            
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r") as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {"accuracy": 0.0}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs TF-IDF transform and outputs prediction probabilities for single text.
        """
        if self.pipeline is None:
            self.load_model()
            
        start_time = time.time()
        
        # Predict label and proba distributions
        prediction = self.pipeline.predict([text])[0]
        proba_list = self.pipeline.predict_proba([text])[0]
        
        # Map class names to their probabilities
        classes = self.pipeline.classes_
        probabilities = {str(c): float(p) for c, p in zip(classes, proba_list)}
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "text": text,
            "sentiment_prediction": str(prediction),
            "probabilities": probabilities,
            "latency_ms": round(latency_ms, 2)
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Runs TF-IDF transform and outputs predictions for a batch of texts.
        """
        if self.pipeline is None:
            self.load_model()
            
        start_time = time.time()
        
        predictions = self.pipeline.predict(texts)
        proba_matrix = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        
        total_latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(texts)
        
        results = []
        for text, pred, proba in zip(texts, predictions, proba_matrix):
            probs = {str(c): float(p) for c, p in zip(classes, proba)}
            results.append({
                "text": text,
                "sentiment_prediction": str(pred),
                "probabilities": probs,
                "latency_ms": round(avg_latency_ms, 2)
            })
            
        return results

# Singleton instance
model_instance = SentimentModel()
