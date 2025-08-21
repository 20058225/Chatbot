# tests/test_dashboard.py
# =======================

## 

import pytest
from unittest.mock import MagicMock
import pages.Dashboard as dash

@pytest.fixture
def mock_db(monkeypatch):
    fake_coll = MagicMock()
    monkeypatch.setattr(dash, "faq", fake_coll)
    monkeypatch.setattr(dash, "knowledge", fake_coll)
    monkeypatch.setattr(dash, "chats", fake_coll)
    monkeypatch.setattr(dash, "monitoring", fake_coll)
    return fake_coll

def test_stats_tab4(monkeypatch, mock_db):
    mock_db.count_documents.return_value = 5
    # Sentiment pipeline
    mock_db.aggregate.side_effect = [
        [{"_id": "Positive", "count": 3}],
        [{
            "avg_response_time": 1.2,
            "fallback_count": 1,
            "thumbs_up_count": 3,
            "thumbs_down_count": 1,
            "unique_users": ["u1", "u2"],
            "total_responses": 4
        }]
    ]
    monkeypatch.setattr(dash, "alt", MagicMock())  # Evita render
    # Simulamos execução manual chamando código da aba 4
    # Nesse caso, como é inline, precisaríamos isolá-lo ou rodar via subprocess

def test_crud_faq_insert(monkeypatch, mock_db):
    dash.st.session_state.clear()
    dash.st.session_state['adding_new_faq'] = True
    monkeypatch.setattr(dash.st, "form_submit_button", lambda label: label == "Save New FAQ")
    monkeypatch.setattr(dash.st, "text_input", lambda *a, **k: "Q1")
    monkeypatch.setattr(dash.st, "text_area", lambda *a, **k: "A1")
    dash.st.form = lambda name: (lambda: None)  # fake context
    mock_db.insert_one.return_value = None
    # Aqui precisaríamos chamar código da aba diretamente ou refatorá-lo
