import joblib
import os
import logging
from functools import lru_cache
from typing import Optional

SENT_MODEL = "../ml/models/sentiment_pipeline.joblib"
PRIO_MODEL = "../ml/models/priority_pipeline.joblib"

@lru_cache(maxsize=2)
def load_sentiment_model() -> Optional[object]:
    try:
        return joblib.load(SENT_MODEL)
    except FileNotFoundError:
        logging.warning(f"Sentiment model not found at {SENT_MODEL}")
        return None
    except Exception as e:
        logging.error(f"Error loading sentiment model: {e}")
        return None

@lru_cache(maxsize=2)
def load_priority_model() -> Optional[object]:
    try:
        return joblib.load(PRIO_MODEL)
    except FileNotFoundError:
        logging.warning(f"Priority model not found at {PRIO_MODEL}")
        return None
    except Exception as e:
        logging.error(f"Error loading priority model: {e}")
        return None

def predict_sentiment(text: str) -> str:
    model = load_sentiment_model()
    if model is None:
        return "unknown"
    return model.predict([text])[0]
