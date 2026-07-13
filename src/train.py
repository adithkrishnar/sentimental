import os
import sys
import glob
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-analysis")
MODEL_REGISTRY_NAME = "SentimentPipeline"

# Small inline dataset used when --sample flag is passed (or SAMPLE_MODE=1)
SAMPLE_DATA = {
    "text": [
        "I love this product, it is amazing!",
        "Fantastic experience, highly recommend.",
        "Absolutely wonderful, best purchase ever.",
        "Great quality and fast shipping.",
        "Very happy with this item.",
        "This is terrible, complete waste of money.",
        "Awful product, broke after one day.",
        "Very disappointed, will not buy again.",
        "Horrible quality, do not buy.",
        "Worst purchase I have ever made.",
        "It's okay, nothing special.",
        "Average product, does the job.",
        "Not bad, not great either.",
        "Decent for the price.",
        "Mediocre, expected more.",
    ],
    "sentiment": [
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "neutral", "neutral", "neutral", "neutral", "neutral",
    ]
}

def train_model(sample_mode: bool = False, nrows: int = None):
    text_col, target_col = "text", "sentiment"

    if sample_mode:
        print("[SAMPLE MODE] Using built-in mini dataset — skipping Kaggle download.")
        df = pd.DataFrame(SAMPLE_DATA)
    else:
        import kagglehub
        print("Downloading abhi8923shriv/sentiment-analysis-dataset from Kaggle...")
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        print(f"Dataset files downloaded to: {path}")

        train_files = glob.glob(os.path.join(path, "**", "*train.csv"), recursive=True)
        if not train_files:
            train_files = glob.glob(os.path.join(path, "*.csv"))
        if not train_files:
            raise FileNotFoundError(f"Could not find any CSV files in path: {path}")

        train_path = train_files[0]
        print(f"Loading dataset from: {train_path}")
        try:
            df = pd.read_csv(train_path, encoding='utf-8', nrows=nrows)
        except UnicodeDecodeError:
            df = pd.read_csv(train_path, encoding='latin-1', nrows=nrows)

        # Detect text and sentiment columns
        cols = df.columns.tolist()
        if 'text' in cols:
            text_col = 'text'
        elif 'selected_text' in cols:
            text_col = 'selected_text'
        else:
            text_col = next((c for c in cols if 'text' in c.lower()), cols[1])
        if 'sentiment' in cols:
            target_col = 'sentiment'
        else:
            target_col = next((c for c in cols if 'sentiment' in c.lower()), cols[-1])
        print(f"Detected columns - Text: '{text_col}', Target: '{target_col}'")

    # Clean
    df = df.dropna(subset=[text_col, target_col])
    df[text_col] = df[text_col].astype(str)
    df[target_col] = df[target_col].astype(str).str.lower().str.strip()
    df = df[df[text_col].str.strip() != ""]

    X = df[text_col]
    y = df[target_col]

    # Train-test split (small sample: no stratify to avoid errors)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    # Build text classification pipeline
    print("Building TF-IDF + Logistic Regression pipeline...")
    tfidf_params = {"max_features": 20000, "ngram_range": (1, 2), "stop_words": "english"}
    lr_params = {"max_iter": 1000, "solver": "lbfgs", "random_state": 42}
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(**lr_params))
    ])
    
    print("Fitting model (this may take a few seconds)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts locally (for FastAPI)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/sentiment_pipeline.joblib")
    print("Saved model pipeline to models/sentiment_pipeline.joblib")
    
    # Build metrics dict
    metrics = {
        "accuracy": round(accuracy, 4),
        "sample_size": len(df)
    }
    for label in pipeline.classes_:
        if label in report:
            metrics[f"f1_{label}"] = round(report[label]["f1-score"], 4)
            metrics[f"precision_{label}"] = round(report[label]["precision"], 4)
            metrics[f"recall_{label}"] = round(report[label]["recall"], 4)

    with open("models/metrics.json", "w") as f:
        json.dump({**metrics, "classes": list(pipeline.classes_)}, f, indent=2)
    print("Saved training metrics to models/metrics.json")

    # ── MLflow Tracking ──────────────────────────────────────────────────────
    print(f"Logging run to MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="TF-IDF+LR") as run:
        # Log hyperparameters
        mlflow.log_param("tfidf_max_features", tfidf_params["max_features"])
        mlflow.log_param("tfidf_ngram_range", str(tfidf_params["ngram_range"]))
        mlflow.log_param("tfidf_stop_words", tfidf_params["stop_words"])
        mlflow.log_param("lr_max_iter", lr_params["max_iter"])
        mlflow.log_param("lr_solver", lr_params["solver"])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("text_column", text_col)
        mlflow.log_param("target_column", target_col)

        # Log metrics
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(key, val)

        # Log local joblib artifact
        mlflow.log_artifact("models/sentiment_pipeline.joblib", artifact_path="model_files")
        mlflow.log_artifact("models/metrics.json", artifact_path="model_files")

        # Log sklearn model + register it in the Model Registry
        signature = infer_signature(X_train.tolist(), y_pred)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sentiment_model",
            signature=signature,
            input_example=X_train.iloc[:3].tolist(),
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        print(f"MLflow run complete. Run ID: {run.info.run_id}")
        print(f"Model registered as: {MODEL_REGISTRY_NAME}")
    # ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "--sample" in sys.argv or os.getenv("SAMPLE_MODE", "0") == "1"
    # Optional: limit rows to reduce memory usage, e.g. NROWS=5000
    nrows_env = os.getenv("NROWS")
    nrows = int(nrows_env) if nrows_env else None
    train_model(sample_mode=sample, nrows=nrows)
