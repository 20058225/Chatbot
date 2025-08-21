# tests/test_monotoring.py
# =======================
## 

import os
import pandas as pd
import mongomock
import pytest
from datetime import datetime, timezone
from unittest.mock import patch
import services.monitoring as monitoring

@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    client = mongomock.MongoClient()
    db = client["test_db"]
    monkeypatch.setattr(monitoring, "db", db)
    monitoring.test_results_col = db["test_results"]
    monitoring.monitoring_col = db["monitoring"]
    monitoring.chats = db["chats"]
    monitoring.monitor_col = db["monitoring"]
    yield db

def test_log_event_info(caplog):
    caplog.set_level("INFO")
    monitoring.log_event("Test message", "info")
    assert "Test message" in caplog.text

def test_log_user_interaction_inserts():
    monitoring.log_user_interaction(
        "user123", "Q?", "A!", "greet", "positive", "high",
        1, 0, False, 1.23
    )
    doc = monitoring.monitor_col.find_one({"user_id": "user123"})
    assert doc is not None
    assert doc["intent_tag"] == "greet"

def test_log_error_inserts():
    try:
        raise ValueError("fail here")
    except Exception as e:
        monitoring.log_error("user456", e)
    doc = monitoring.monitor_col.find_one({"user_id": "user456"})
    assert doc is not None
    assert doc["error_type"] == "ValueError"

def test_generate_report_empty():
    assert monitoring.generate_report() == "No interactions logged yet."

def test_generate_report_with_data():
    monitoring.monitor_col.insert_one({
        "user_id": "u1", "intent_tag": "test", "response_time": 1.0,
        "is_fallback": False, "thumbs_up": True, "thumbs_down": False
    })
    report = monitoring.generate_report()
    assert "Total Responses" in report

def test_log_execution_creates_docs_and_csv(tmp_path):
    monitoring.LOG_DIR = tmp_path
    monitoring.log_execution(1.0, 0.9, 2.0, 0.8, execution_type="test")
    assert monitoring.test_results_col.count_documents({}) == 2
    assert any(f.suffix == ".csv" for f in tmp_path.iterdir())

def test_load_logs_returns_df():
    monitoring.test_results_col.insert_one({
        "timestamp": datetime.now(timezone.utc),
        "model": "gpt-2",
        "execution_time": 1.0,
        "score": 0.9,
        "log_source": "simulation"
    })
    df = monitoring.load_logs("test")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
