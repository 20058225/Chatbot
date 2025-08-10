# services/ml.py

import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Caminho base dos modelos
MODELS_DIR = Path("ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Variáveis para lazy load
_priority_model = None
_sentiment_model = None

# =========================================================
# Lazy Load Models
# =========================================================

def load_priority_model():
    global _priority_model
    if _priority_model is None:
        path = MODELS_DIR / "priority_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Priority model not found at {path}")
        _priority_model = joblib.load(path)
    return _priority_model

def load_sentiment_model():
    global _sentiment_model
    if _sentiment_model is None:
        path = MODELS_DIR / "sentiment_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Sentiment model not found at {path}")
        _sentiment_model = joblib.load(path)
    return _sentiment_model

# =========================================================
# Prediction Wrappers
# =========================================================

def predict_priority(texts):
    model = load_priority_model()
    return model.predict(texts)

def predict_sentiment(texts):
    model = load_sentiment_model()
    return model.predict(texts)

# =========================================================
# Treinamento e Salvamento
# =========================================================

def train_and_save_models_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "text" in df.columns:
        texts = df["text"]
    elif "description" in df.columns:
        texts = df["description"]
    else:
        raise ValueError("CSV must contain 'text' or 'description' column")

    # --- Priority Model ---
    if "priority" in df.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, df["priority"], test_size=0.2, random_state=42
        )
        priority_model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000)
        )
        priority_model.fit(X_train, y_train)
        joblib.dump(priority_model, MODELS_DIR / "priority_pipeline.joblib")

    # --- Sentiment Model ---
    if "sentiment" in df.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, df["sentiment"], test_size=0.2, random_state=42
        )
        sentiment_model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(max_iter=1000)
        )
        sentiment_model.fit(X_train, y_train)
        joblib.dump(sentiment_model, MODELS_DIR / "sentiment_pipeline.joblib")

def train_and_save_kmeans_from_csv(csv_path, n_clusters=3):
    df = pd.read_csv(csv_path)

    if "text" in df.columns:
        texts = df["text"]
    elif "description" in df.columns:
        texts = df["description"]
    else:
        raise ValueError("CSV must contain 'text' or 'description' column'")
    
    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_vectors)
    joblib.dump((kmeans, vectorizer), MODELS_DIR / "kmeans_model.joblib")
