# services/nlp_utils.py

import joblib
from ml.sentiment_model import MODEL_PATH as SENT_MODEL
from ml.priority_classifier import MODEL_PATH as PRIO_MODEL

# load once on import
_sentiment_pipe = joblib.load(SENT_MODEL)
_priority_pipe  = joblib.load(PRIO_MODEL)

def predict_sentiment(text: str) -> str:
    """Return 'positive', 'neutral' or 'negative'."""
    return _sentiment_pipe.predict([text])[0]

def predict_priority(text: str) -> str:
    """Return 'High', 'Medium' or 'Low'."""
    return _priority_pipe.predict([text])[0]
