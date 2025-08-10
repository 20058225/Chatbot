# tests/test_services.py
# ======================
import pytest
from services.db import append_message, update_message_feedback, find_known_answer, get_chats_collection
from datetime import datetime

def test_save_chat_message():
    append_message("test_session", {"sender": "user", "text": "hello", "timestamp": datetime.utcnow()})
    chat = get_chats_collection().find_one({"session_id": "test_session"})
    assert any(msg["text"] == "hello" for msg in chat["messages"])

def test_update_message_feedback():
    chat = get_chats_collection().find_one({"session_id": "test_session"})
    update_message_feedback(chat["_id"], 0, "positivo")
    updated = get_chats_collection().find_one({"_id": chat["_id"]})
    assert updated["messages"][0]["feedback"] == "positivo"

def test_find_known_answer_none():
    assert find_known_answer("nonexistent question") is None
