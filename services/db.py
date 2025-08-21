# services/db.py

from bson import ObjectId
from datetime import datetime, timezone
from services.mongo import db

# Coleções
chats_col = db["chat"]
unanswered_col = db["unanswered"]
requests_col = db["requests"]
default_chat_col = db["default_chat"]
users_col = db["users"]
monitoring_col = db["monitoring"]
test_results_col = db["test_results"]

# =========================================================
# Funções de Chat
# =========================================================

def save_chat_message(session_id, role, content, feedback=None):
    chat_doc = chats_col.find_one({"session_id": session_id})
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc),
        "feedback": feedback or ""
    }

    if chat_doc:
        chats_col.update_one(
            {"_id": chat_doc["_id"]},
            {"$push": {"messages": message}}
        )
    else:
        chats_col.insert_one({
            "session_id": session_id,
            "start_time": datetime.now(timezone.utc),
            "messages": [message]
        })

def update_chat_message(chat_id, message_index, new_content):
    chat_doc = chats_col.find_one({"_id": ObjectId(chat_id)})
    if not chat_doc:
        return False

    messages = chat_doc.get("messages", [])
    if 0 <= message_index < len(messages):
        messages[message_index]["content"] = new_content
        chats_col.update_one({"_id": ObjectId(chat_id)}, {"$set": {"messages": messages}})
        return True
    return False

def update_message_feedback(chat_id, message_index, feedback):
    chat_doc = chats_col.find_one({"_id": ObjectId(chat_id)})
    if not chat_doc:
        return False

    messages = chat_doc.get("messages", [])
    if 0 <= message_index < len(messages):
        messages[message_index]["feedback"] = feedback
        chats_col.update_one({"_id": ObjectId(chat_id)}, {"$set": {"messages": messages}})
        return True
    return False

def get_all_chats():
    return list(chats_col.find().sort("start_time", -1))

# =========================================================
# Funções de FAQ / Base de Conhecimento
# =========================================================

def find_known_answer(user_input):
    doc = requests_col.find_one({"question": {"$regex": user_input, "$options": "i"}})
    if doc:
        return doc.get("answer")
    return None

# =========================================================
# Funções para Mensagens Não Respondidas
# =========================================================

def save_unanswered(session_id, question):
    unanswered_col.insert_one({
        "session_id": session_id,
        "question": question,
        "timestamp": datetime.now(timezone.utc)
    })

def save_embedding_evaluation_results(results: dict, log_source="embedding_evaluation", extra_meta=None):
    """
    Salva os resultados da avaliação de embeddings no MongoDB 'test_results'.
    """
    timestamp_now = datetime.now(timezone.utc)

    for model_name, metrics in results.items():
        doc = {
            "timestamp": timestamp_now,
            "model": model_name.lower(),
            "log_source": log_source,
            "accuracy": metrics["classification"]["accuracy"],
            "precision": metrics["classification"]["precision"],
            "recall": metrics["classification"]["recall"],
            "f1_score": metrics["classification"]["f1_score"],
            "embedding_train_time": metrics["classification"]["embedding_train_time_sec"],
            "embedding_test_time": metrics["classification"]["embedding_test_time_sec"],
            "silhouette_score": metrics["clustering"]["silhouette_score"]
        }
        if extra_meta:
            doc.update(extra_meta)
        test_results_col.insert_one(doc)
