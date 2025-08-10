# services/mongo.py

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")
mongo = os.getenv("API_MONGO")
if not mongo:
    raise ValueError("API_MONGO not set! Check your .env loading.")

client = MongoClient(mongo)
db = client["chatbotDB"]
