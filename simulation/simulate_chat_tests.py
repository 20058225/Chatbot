# simulation/simulate_chat_tests.py

import argparse
import time
import pandas as pd
import uuid
import csv
from datetime import datetime, timezone
import torch
import logging
from pathlib import Path

# ==== CONFIG MONGO ====
from services.mongo import db
from services.monitoring import log_execution

# ==== CONFIG LOGGING ====
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

test_results_col = db["test_results"]
monitoring_col = db["monitoring"]

faq = db["faq"]
default_chat = db["default_chat"]
knowledge = db["knowledge"]

def log_event(event_type, details, status="success", log_source="simulation"):
    monitoring_col.insert_one({
        "event": event_type,
        "details": details,
        "status": status,
        "log_source": log_source,
        "timestamp": datetime.now(timezone.utc)
    })


def load_test_queries(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_bert_best_match(query, kb_entries):
    from pages.Chatbot import get_bert_embeddings
    query_emb = get_bert_embeddings(query)
    best_score = -1
    best_answer = "No match found."
    for entry in kb_entries:
        entry_text = f"{entry.get('question', '')} {entry.get('answer', '')}"
        entry_emb = get_bert_embeddings(entry_text)
        score = torch.cosine_similarity(
            torch.tensor(query_emb), 
            torch.tensor(entry_emb), 
            dim=0
        ).item()
        if score > best_score:
            best_score = score
            best_answer = entry.get("answer", "No answer available.")
    return best_answer, best_score


def simulate_test(model_name, questions, execution_type="test"):
    start_time = time.time()
    
    # Simulate model processing (replace with actual model call)
    time.sleep(0.1 * len(questions))  # Simulated delay
    score = round(0.7 + (0.3 * len(questions) / 100), 4)  # Fake score

    execution_time = round(time.time() - start_time, 4)

    # Log execution
    log_execution(model=model_name, execution_time=execution_time, score=score, execution_type=execution_type)

    return {"model": model_name, "execution_time": execution_time, "score": score}


# ==== MAIN TEST LOOP ====
def run_tests(questions_file="data/questions.txt"):
    from pages.Chatbot import generate_gpt2_reply, get_bert_embeddings

    TEST_QUERIES = load_test_queries(questions_file)
    logging.info(f"Starting automated chatbot tests with {len(TEST_QUERIES)} queries...")
    user_id = f"test_{uuid.uuid4().hex[:8]}"

    # Carrega KB real do Mongo
    kb_entries = []
    # FAQ
    for doc in db["faq"].find({}, {"_id": 0, "question": 1, "answer": 1}):
        kb_entries.append({
            "question": doc.get("question", ""), 
            "answer": doc.get("answer", "")})
    # default_chat
    intent_doc = db["default_chat"].find_one({
        "intents": {"$exists": True}}, 
        {"_id": 0, "intents": 1})
    if intent_doc:
        for intent in intent_doc.get("intents", []):
            for pattern in intent.get("patterns", []):
                for response in intent.get("responses", []):
                    kb_entries.append({
                        "question": pattern, 
                        "answer": response})
    # knowledge
    for doc in db["knowledge"].find({}, {"_id": 0, "title": 1, "content": 1}):
        kb_entries.append({
            "question": doc.get("title", ""), 
            "answer": doc.get("content", "")})

    results = []

    total_queries = len(TEST_QUERIES)
    for i, query in enumerate(TEST_QUERIES, start=1):

        # --- Barra de progresso no terminal ---
        progress = int((i / total_queries) * 100)
        logging.info(f"Progresso: {progress}% ({i}/{total_queries})")

        # --- Exibir no Streamlit se disponível ---
        try:
            import streamlit as st
            st.progress(progress)
        except ImportError:
            pass  # Ignora se não estiver rodando no Streamlit

        # --- GPT-2 Test ---
        start_time = time.time()
        gpt2_response = generate_gpt2_reply(query)
        gpt2_time = round(time.time() - start_time, 3)

        # --- BERT Test ---
        start_time = time.time()
        bert_response, bert_score = get_bert_best_match(query, kb_entries)
        bert_time = round(time.time() - start_time, 3)

        try:
            gpt2_emb = get_bert_embeddings(gpt2_response)
            bert_ref_emb = get_bert_embeddings(bert_response)
            gpt2_score = torch.cosine_similarity(
                torch.tensor(gpt2_emb),
                torch.tensor(bert_ref_emb),
                dim=0
            ).item()
        except Exception:
            gpt2_score = None

        # Save to MongoDB
        test_results_col.insert_one({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "response": gpt2_response,
            "score": gpt2_score,
            "execution_time": gpt2_time,
            "model": "gpt-2",
            "log_source": "simulation"
        })

        test_results_col.insert_one({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "response": bert_response,
            "score": bert_score,
            "execution_time": bert_time,
            "model": "bert",
            "log_source": "simulation"
        })

        results.append({
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "query": query,
            "gpt2_response": gpt2_response,
            "gpt2_score": gpt2_score,
            "bert_response": bert_response,
            "bert_score": bert_score,
            "gpt2_time": gpt2_time,
            "bert_time": bert_time
        })
        
        log_event("test_case", {"query": query}, log_source="simulation")
                
        logging.info(f"Query: {query}")
        logging.info(f"GPT-2 [{gpt2_time}s, score={gpt2_score}]: {gpt2_response}")
        logging.info(f"BERT [{bert_time}s, score={bert_score:.3f}]: {bert_response}")
        logging.info("-" * 50)

    # Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"simulation/result/test_results-{timestamp}.csv"
    Path("simulation/result").mkdir(parents=True, exist_ok=True)
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "user_id", "timestamp", "query",
            "gpt2_response", "gpt2_score",
            "bert_response", "bert_score",
            "gpt2_time", "bert_time"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    logging.info(f"✅ Test results saved to {csv_filename}")
    logging.info(f"✅ Test logs saved to MongoDB collections 'test_results', and 'monitoring'")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/questions.txt", help="Path to questions file")
    args = parser.parse_args()
    run_tests(args.file)

    test_questions = ["How do I reset my password?", "Where is my invoice?", "Can I change my subscription?"]

    results = []
    results.append(simulate_test("BERT", test_questions, "test"))
    results.append(simulate_test("GPT2", test_questions, "test"))

    df_results = pd.DataFrame(results)
    print(df_results)