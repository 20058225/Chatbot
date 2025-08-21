# services/monitoring.py
import logging
import os
import bson
import pandas as pd
from datetime import datetime, timezone
from services.mongo import db

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# Log directory and file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chatbot_monitor.log") 

test_results_col = db["test_results"]
monitoring_col = db["monitoring"]
chats = db["chats"]

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_event(message: str, level: str = "info"):
    """Write a log entry with the given level.
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
    """
    getattr(logging, level.lower(), logging.info)(message)


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
        "timestamp": datetime.now(timezone.utc)
    }
    monitoring_col.insert_one(doc)

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
        "timestamp": datetime.now(timezone.utc)
    }
    monitoring_col.insert_one(doc)
    logging.error(f"user_id={user_id} | Exception type={err_type}: {error}")


def generate_report():
    total_responses = monitoring_col.count_documents({"intent_tag": {"$exists": True}})
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
    result_list = list(monitoring_col.aggregate(pipeline))
    if not result_list:
        return "No interactions logged yet."

    result = result_list[0]

    fallback_rate = result["fallback_count"] / total_responses
    total_feedback = result["thumbs_up_count"] + result["thumbs_down_count"]
    satisfaction_score = ((result["thumbs_up_count"] - result["thumbs_down_count"]) / total_feedback) if total_feedback > 0 else 0
    unique_users = len(result["unique_users"])

    error_count = monitoring_col.count_documents({"error_type": {"$exists": True}})

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


def log_execution(
    gpt2_time,
    gpt2_score,
    bert_time,
    bert_score,
    execution_type="test",
    user_id=None,
    query=None,
    gpt2_response=None,
    bert_response=None
):
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = os.path.join(LOG_DIR, f"{execution_type}_log.csv")
    all_log_file = os.path.join(LOG_DIR, "all_log.csv")

    now_dt = datetime.now(timezone.utc)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    log_entry_gpt2 = {
        "timestamp": now_dt,
        "model": "gpt-2",
        "execution_time": gpt2_time,
        "score": gpt2_score,
        "execution_type": execution_type,
        "log_source": "simulation"
    }
    if user_id: 
        log_entry_gpt2["user_id"] = user_id
    if query: 
        log_entry_gpt2["query"] = query
    if gpt2_response: 
        log_entry_gpt2["response"] = gpt2_response

    log_entry_bert = {
        "timestamp": now_dt,
        "model": "bert",
        "execution_time": bert_time,
        "score": bert_score,
        "execution_type": execution_type,
        "log_source": "simulation"
    }
    if user_id: 
        log_entry_bert["user_id"] = user_id
    if query: 
        log_entry_bert["query"] = query
    if bert_response: 
        log_entry_bert["response"] = bert_response

    # Salva no MongoDB
    test_results_col.insert_one(log_entry_gpt2)
    test_results_col.insert_one(log_entry_bert)

    df = pd.DataFrame([log_entry_gpt2, log_entry_bert])
    for file in [filename, all_log_file]:
        if os.path.exists(file):
            df.to_csv(file, mode="a", header=False, index=False)
        else:
            df.to_csv(file, index=False)
    
    logging.info(
        f"Execução registrada: GPT-2 (t={gpt2_time}s, score={gpt2_score}) | "
        f"BERT (t={bert_time}s, score={bert_score}) | query={query}"
    )


def load_logs(log_type="all"):
    """Carrega logs do Mongo e retorna DataFrame com dados já tratados para o Streamlit."""

    if log_type == "train":
        cursor = monitoring_col.find({"event": "train_models"})
        data = list(cursor)
    elif log_type == "test":
        cursor = test_results_col.find({"log_source": "simulation"})
        data = list(cursor)
    else:  # all
        train_logs = list(monitoring_col.find({"event": "train_models"}))
        test_logs = list(test_results_col.find({"log_source": "simulation"}))
        data = train_logs + test_logs

    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)

    # 🔹 1. Converter ObjectId para string
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, bson.ObjectId) else x))

    # 🔹 2. Converter timestamp para datetime se existir
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass

    # 🔹 3. Garantir que colunas numéricas sejam numéricas
    for col in ["execution_time", "score", "gpt2_time", "bert_time", "bert_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df