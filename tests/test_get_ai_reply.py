import pytest
from unittest.mock import patch, MagicMock
from pages.Chatbot import get_ai_reply 
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@patch("pages.Chatbot.faq.find")
@patch("pages.Chatbot.default_chat.find_one")
@patch("pages.Chatbot.db")
@patch("pages.Chatbot.model.start_chat")
def test_get_ai_reply_mocked(
    mock_start_chat,
    mock_db,
    mock_find_one,
    mock_faq_find
):
    # 📚 Mock the MongoDB context data
    mock_faq_find.return_value.limit.return_value = [
        {"question": "How to reset password?", "answer": "Click on forgot password"},
    ]

    mock_find_one.return_value = {
        "intents": [
            {"tag": "greeting", "patterns": ["hello"], "responses": ["Hi there!"]},
        ]
    }

    mock_db["knowledge"].find.return_value.limit.return_value = [
        {"title": "VPN Setup", "content": "To set up VPN, go to Settings..."}
    ]

    mock_chat = MagicMock()
    mock_chat.send_message.return_value.text = "Hello! How can I help you today?"
    mock_start_chat.return_value = mock_chat

    response = get_ai_reply("hi")

    assert "Hello!" in response
    mock_chat.send_message.assert_called_once()
