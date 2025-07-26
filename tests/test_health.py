import streamlit as st
from pages import Chatbot
from ml.priority import classify_priority
from services.mongo import db

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    assert callable(classify_priority)
    assert "users" in db.list_collection_names()
