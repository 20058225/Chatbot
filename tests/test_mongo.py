import pytest
from services.mongo import db
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_db_connection():
    assert db is not None
    assert 'users' in db.list_collection_names()

def test_user_insert_and_find():
    users = db['users']
    test_doc = {'name': 'Test', 'email': 'test@example.com'}
    ins = users.insert_one(test_doc)
    fetched = users.find_one({'_id': ins.inserted_id})
    assert fetched['email'] == 'test@example.com'
    users.delete_one({'_id': ins.inserted_id})

def test_faq_schema():
    faq = db['faq']
    doc = {"question": "What is VPN?", "answer": "Virtual Private Network"}
    inserted = faq.insert_one(doc)
    fetched = faq.find_one({'_id': inserted.inserted_id})
    assert 'question' in fetched and 'answer' in fetched
    faq.delete_one({'_id': inserted.inserted_id})

