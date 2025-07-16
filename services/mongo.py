import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

mongo = os.getenv("API_MONGO")

client = MongoClient(mongo)
db = client["chatbotDB"]
