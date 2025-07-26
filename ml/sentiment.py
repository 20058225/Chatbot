import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

DATA_PATH = "data/tickets.csv"
MODEL_PATH = "ml/models/sentiment_pipeline.joblib"

try:
    model = load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: could not load sentiment model: {e}")

def classify_sentiment(text):
    if not text:
        return "neutral"
    if model is None:
        raise RuntimeError("Modelo de sentimento não carregado")
    pred = model.predict([text])[0]
    print(f"Classify sentiment input: {text} -> prediction: {pred}")
    return pred.lower()

def train_sentiment():
    df = pd.read_csv(DATA_PATH)
    # we assume df has a 'sentiment' column
    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["sentiment"], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    print("Train accuracy:", pipeline.score(X_train, y_train))
    print("Test accuracy: ", pipeline.score(X_test, y_test))

    # save pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Sentiment model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_sentiment()
