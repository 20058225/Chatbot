# tests/test_chatbot.py
# ======================

## pytest -v --maxfail=1 --disable-warnings

import pytest
import types
import numpy as np
import torch
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pages.Chatbot as chatbot


@pytest.fixture
def mock_db(monkeypatch):
    """Mocka todas as coleções do Mongo."""
    fake_coll = MagicMock()
    monkeypatch.setattr(chatbot, "users", fake_coll)
    monkeypatch.setattr(chatbot, "chats", fake_coll)
    monkeypatch.setattr(chatbot, "faq", fake_coll)
    monkeypatch.setattr(chatbot, "knowledge", fake_coll)
    monkeypatch.setattr(chatbot, "unanswered", fake_coll)
    monkeypatch.setattr(chatbot, "default_chat", fake_coll)
    monkeypatch.setattr(chatbot, "monitoring_col", fake_coll)
    monkeypatch.setattr(chatbot.db, "feedback", fake_coll)
    return fake_coll


def test_cosine_similarity():
    a = np.array([1, 0])
    b = np.array([1, 0])
    # comparação correta com pytest.approx
    assert chatbot.cosine_similarity(a, b) == pytest.approx(1.0)

    c = np.array([0, 1])
    assert abs(chatbot.cosine_similarity(a, c)) < 1e-6


def test_is_similar():
    assert chatbot.is_similar("Hello World", "hello world")
    assert not chatbot.is_similar("Hello", None)
    assert not chatbot.is_similar("abc", "xyz")


def test_generate_chat_id_format():
    cid = chatbot.generate_chat_id()
    assert "-" in cid
    assert len(cid.split("-")[1]) == 8


def test_get_bert_embeddings_mock(monkeypatch):
    monkeypatch.setattr(
        chatbot, 
        "bert_tokenizer", 
        MagicMock(return_value={"input_ids": torch.tensor([[1]])})
    )

    class DummyOutput:
        last_hidden_state = torch.tensor([[[1., 2., 3.]]])

    dummy_model = MagicMock(return_value=DummyOutput())
    dummy_model.__enter__ = lambda s: s
    dummy_model.__exit__ = lambda s, *a: None
    monkeypatch.setattr(chatbot, "bert_model", dummy_model)

    class DummyContext:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    with patch("torch.no_grad", return_value=DummyContext()):
        result = chatbot.get_bert_embeddings("test")

    assert isinstance(result, np.ndarray)


from unittest.mock import MagicMock, patch

class DummyContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

def test_generate_gpt2_reply_mock(monkeypatch):
    monkeypatch.setattr(chatbot, "gpt2_tokenizer", MagicMock(
        return_value={"input_ids": [[1]]},   # lista aninhada para simular batch
        decode=lambda x, skip_special_tokens=True: "Hello"
    ))
    fake_model = MagicMock()
    fake_model.generate.return_value = [[0, 1]]
    monkeypatch.setattr(chatbot, "gpt2_model", fake_model)

    with patch("torch.no_grad", return_value=DummyContext()):
        reply = chatbot.generate_gpt2_reply("Hi")

    assert reply == "Hello"



def test_predict_sentiment_with_text(monkeypatch):
    monkeypatch.setattr(chatbot, "predict_sentiment", lambda x: ["Positive"])
    result = chatbot.predict_sentiment_with_text("ok")
    assert result == "Positive"

    # Test fallback
    monkeypatch.setattr(chatbot, "predict_sentiment", lambda x: 1 / 0)
    assert chatbot.predict_sentiment_with_text("fail") == "unknown"


def test_predict_priority_with_text(monkeypatch):
    monkeypatch.setattr(chatbot, "predict_priority", lambda x: ["High"])
    result = chatbot.predict_priority_with_text("ok")
    assert result == "High"

    # Fallback
    monkeypatch.setattr(chatbot, "predict_priority", lambda x: 1 / 0)
    assert chatbot.predict_priority_with_text("fail") == "Low"


def test_find_default_answer(monkeypatch, mock_db):
    mock_db.find_one.return_value = {"intents": [
        {"patterns": ["hello"], "responses": ["hi"], "tag": "greet"}
    ]}
    monkeypatch.setattr(chatbot, "is_similar", lambda a, b, threshold=0.85: True)
    ans, tag = chatbot.find_default_answer("hello")
    assert ans == "hi"
    assert tag == "greet"


def test_find_known_answer(monkeypatch, mock_db):
    mock_db.find_one.return_value = {"answer": "42"}
    assert chatbot.find_known_answer("question") == "42"

    mock_db.find_one.return_value = None
    mock_db.find.return_value.limit.return_value = [
        {"question": "answer me", "answer": "yes"}
    ]
    monkeypatch.setattr(chatbot, "SequenceMatcher", lambda *a, **k: types.SimpleNamespace(ratio=lambda: 0.8))
    assert chatbot.find_known_answer("answer me") == "yes"


def test_find_knowledge_answer(mock_db):
    mock_db.find_one.return_value = {"content": "info"}
    assert chatbot.find_knowledge_answer("help") == "info"


def test_log_event(mock_db):
    chatbot.log_event("test", {"msg": "ok"})
    mock_db.insert_one.assert_called()


def test_save_chat_message(mock_db, monkeypatch):
    monkeypatch.setitem(chatbot.st.session_state, "session_id", None)
    monkeypatch.setitem(chatbot.st.session_state, "user", {"email": "a@b.com"})

    sid, mid, msg = chatbot.save_chat_message({"email": "a@b.com"}, "Q", "A")
    assert isinstance(sid, str)


def test_send_email_mock(monkeypatch):
    fake_smtp = MagicMock()
    fake_smtp.send = MagicMock()
    monkeypatch.setattr(chatbot.yagmail, "SMTP", lambda a, b: fake_smtp)
    chatbot.send_email("to@test.com", "subj", "body")
    fake_smtp.send.assert_called()


def test_handle_unanswered(monkeypatch, mock_db):
    monkeypatch.setattr(chatbot, "send_email", lambda *a, **k: None)
    chatbot.handle_unanswered({"email": "a@b.com", "name": "Foo"}, "Q")


def test_handle_feedback(monkeypatch, mock_db):
    monkeypatch.setattr(chatbot, "send_email", lambda *a, **k: None)
    chatbot.handle_feedback({"email": "a@b.com", "name": "Foo"}, "Q", "A", "TID", "Great!", liked=True)


def test_get_chat_topic():
    msgs = [{"intent_tag": "billing"}, {"intent_tag": "billing"}]
    assert chatbot.get_chat_topic(msgs) == "billing"

    msgs = [{"question": "reset password"}]
    assert "reset" in chatbot.get_chat_topic(msgs)

    assert chatbot.get_chat_topic([]) == "No topic"


def test_reload_chat_history(mock_db, monkeypatch):
    monkeypatch.setitem(chatbot.st.session_state, "session_id", "123")
    monkeypatch.setitem(chatbot.st.session_state, "chat_loaded_for_session", None)
    mock_db.find_one.return_value = {"messages": [{"q": "a"}]}
    chatbot.reload_chat_history(force=True)
    assert "chat_history" in chatbot.st.session_state
