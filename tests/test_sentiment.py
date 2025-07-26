import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.sentiment import classify_sentiment

def test_sentiment_positive():
    result = classify_sentiment("Thank you so much! That was very helpful.")
    assert result.lower() == "positive"

def test_sentiment_negative():
    result = classify_sentiment("This is not working at all. Very frustrating.")
    assert result.lower() == "negative"

def test_sentiment_neutral():
    result = classify_sentiment("Please send me the installation guide.")
    assert result.lower() == "neutral"

def test_sentiment_empty():
    result = classify_sentiment("")
    assert result.lower() == "neutral"

def test_sentiment_none():
    result = classify_sentiment(None)
    assert result.lower() == "neutral"
