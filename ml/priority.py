import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = "data/tickets.csv"
MODEL_PATH = "ml/models/priority_pipeline.joblib"


def classify_priority(text):
    if not text:
        return "low"

    text = text.lower()

    if "urgent" in text or "asap" in text or "critical" in text:
        return "high"
    if "how to" in text or "can you explain" in text:
        return "low"

    # Fallback to model prediction
    from joblib import load
    model = load("ml/models/priority_pipeline.joblib")
    pred = model.predict([text])[0].lower()
    print(f"Classify priority input: '{text}' -> prediction: '{pred}'")
    return pred


def train_priority():
    df = pd.read_csv(DATA_PATH)
    # priority column must be categorical: High, Medium, Low
    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["priority"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)
    print("Train accuracy:", pipeline.score(X_train, y_train))
    print("Test accuracy: ", pipeline.score(X_test, y_test))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Priority model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_priority()