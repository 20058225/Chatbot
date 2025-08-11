# services/monitoring.py
import logging
import os
from datetime import datetime
import time

from services.mongo import db

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Log directory and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chatbot_monitor.log") 

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_event(message: str, level: str = "info"):
    """Write a log entry with the given level."""
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    elif level == "debug":
        logging.debug(message)
    else:
        logging.info(message)

def get_log_file_path():
    """Return the absolute path of the log file."""
    return os.path.abspath(LOG_FILE)

def read_logs():
    """Read the log file and return its contents."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return f.read()
    return "No logs found."

MODEL_VERSION = "GPT-4o-mini-v1" 

chats = db["chats"]
monitor_col = db["monitoring"]

def log_user_interaction(user_id, question, answer, intent_tag, sentiment, priority, thumbs_up, thumbs_down, is_fallback, response_time):
    doc = {
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "intent_tag": intent_tag,
        "sentiment": sentiment,
        "priority": priority,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "is_fallback": is_fallback,
        "response_time": response_time,
        "model_version": MODEL_VERSION,
        "timestamp": time.time()
    }
    monitor_col.insert_one(doc)

    logging.info(
        f"user_id={user_id} | intent={intent_tag} | priority={priority} | sentiment={sentiment} | "
        f"thumbs_up={thumbs_up} | thumbs_down={thumbs_down} | fallback={is_fallback} | response_time={response_time:.3f}s\n"
        f"Q: {question}\nA: {answer}\nmodel_version={MODEL_VERSION}"
    )

def log_error(user_id, error):
    err_type = type(error).__name__
    doc = {
        "user_id": user_id,
        "error_type": err_type,
        "error_msg": str(error),
        "timestamp": time.time()
    }
    monitor_col.insert_one(doc)
    logging.error(f"user_id={user_id} | Exception type={err_type}: {error}")

def generate_report():
    total_responses = monitor_col.count_documents({"intent_tag": {"$exists": True}})
    if total_responses == 0:
        return "No interactions logged yet."

    pipeline = [
        {"$match": {"intent_tag": {"$exists": True}}},
        {
            "$group": {
                "_id": None,
                "avg_response_time": {"$avg": "$response_time"},
                "fallback_count": {"$sum": {"$cond": ["$is_fallback", 1, 0]}},
                "thumbs_up_count": {"$sum": {"$cond": ["$thumbs_up", 1, 0]}},
                "thumbs_down_count": {"$sum": {"$cond": ["$thumbs_down", 1, 0]}},
                "unique_users": {"$addToSet": "$user_id"},
            }
        }
    ]
    result = list(monitor_col.aggregate(pipeline))[0]

    fallback_rate = result["fallback_count"] / total_responses
    total_feedback = result["thumbs_up_count"] + result["thumbs_down_count"]
    satisfaction_score = ((result["thumbs_up_count"] - result["thumbs_down_count"]) / total_feedback) if total_feedback > 0 else 0
    unique_users = len(result["unique_users"])

    error_count = monitor_col.count_documents({"error_type": {"$exists": True}})

    report = f"""
Chatbot Monitoring Report:
--------------------------
Total Responses: {total_responses}
Unique Users: {unique_users}
Avg. AI Response Time: {result['avg_response_time']:.3f} seconds
Fallback Rate: {fallback_rate:.2%}
Satisfaction Score (thumbs up minus down): {satisfaction_score:.2f}
Total Errors: {error_count}
"""
    return report
