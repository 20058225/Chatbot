# services/embeddings.py
# ===============================
## python -m services.embeddings --dataset hf
## python -m services.embeddings --dataset local
## python -m services.embeddings --dataset questions

import argparse
import logging
import pandas as pd
import json
import time
import os
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

from services.embeddings import get_bert_embeddings, get_gpt2_embeddings, get_sbert_embeddings
from services.evaluation import run_full_evaluation
from services.mongo import db

logging.basicConfig(level=logging.INFO)

# ==== Funções para carregar datasets ====
def load_hf_dataset():
    logging.info("📥 Baixando dataset 'Tobi-Bueck/customer-support-tickets'...")
    dataset_hf = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = pd.DataFrame(dataset_hf["train"])
    logging.info(f"✅ Dataset carregado com {len(df)} linhas e colunas: {df.columns.tolist()}")
    return df[["ticket_text", "category"]].dropna(), "ticket_text", "category"

def load_local_dataset():
    logging.info("📂 Carregando dataset local 'data/train_model.csv'...")
    df = pd.read_csv("data/train_model.csv")
    logging.info(f"✅ Dataset local carregado com {len(df)} linhas e colunas: {df.columns.tolist()}")
    return df[["description", "sentiment"]].dropna(), "description", "sentiment"

def load_questions_dataset():
    logging.info("📝 Carregando perguntas de 'data/questions.txt'...")
    with open("data/questions.txt", "r", encoding="utf-8") as f:
        questions = [q.strip().strip(",") for q in f.readlines() if q.strip()]
    logging.info(f"✅ {len(questions)} perguntas carregadas.")
    return questions

# ==== Avaliação principal ====
def evaluate_with_labels(df, TEXT_COLUMN, LABEL_COLUMN, n_clusters, dataset_source):
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN].tolist(),
        df[LABEL_COLUMN].tolist(),
        test_size=0.5,
        random_state=42,
        stratify=df[LABEL_COLUMN]
    )

    all_texts = df[TEXT_COLUMN].tolist()

    embed_fns = {
        "BERT": get_bert_embeddings,
        "GPT-2": get_gpt2_embeddings,
        "SBERT": get_sbert_embeddings
    }

    dataset_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "texts": all_texts
    }

    logging.info("🚀 Iniciando avaliação...")
    results = run_full_evaluation(
        embed_fns=embed_fns,
        dataset=dataset_dict,
        classifier=LogisticRegression(max_iter=1000),
        n_clusters=n_clusters
    )

    save_results(results, dataset_source)

def evaluate_questions_only(questions):
    embed_fns = {
        "BERT": get_bert_embeddings,
        "GPT-2": get_gpt2_embeddings,
        "SBERT": get_sbert_embeddings
    }

    results = {}
    for model_name, fn in embed_fns.items():
        logging.info(f"⏳ Gerando embeddings para {model_name}...")
        start_time = time.time()
        _ = fn(questions)  # não vamos armazenar para economizar memória
        elapsed = time.time() - start_time
        results[model_name] = {"embedding_time_sec": elapsed}
        logging.info(f"✅ {model_name} -> {elapsed:.2f} segundos")

    save_results(results, "questions", classification=False)

# ==== Salvar resultados em arquivo e MongoDB ====
def save_results(results, dataset_source, classification=True):
    os.makedirs("results", exist_ok=True) 

    results_path_json = f"results/{dataset_source}.json"
    results_path_csv = f"results/{dataset_source}.csv"

    def to_serializable(val):
        if isinstance(val, (np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.int32, np.int64)):
            return int(val)
        return val

    # Salva JSON com conversão
    with open(results_path_json, "w") as f:
        json.dump(results, f, indent=4, default=to_serializable)

    # Achatar métricas
    flat_results = []
    for model_name, metrics in results.items():
        row = {"model": model_name.lower()}

        if classification:
            # Classificação (médias + std)
            for metric_key in [
                "accuracy", "precision", "recall", "f1_score",
                "embedding_train_time_sec", "embedding_test_time_sec"
            ]:
                val = metrics["classification"].get(metric_key)
                if isinstance(val, dict):
                    row[metric_key] = to_serializable(val.get("mean"))
                    row[f"{metric_key}_std"] = to_serializable(val.get("std"))
                else:
                    row[metric_key] = to_serializable(val)

            # Clustering (silhouette)
            sil_val = metrics.get("clustering", {}).get("silhouette_score", None)
            if isinstance(sil_val, dict):
                row["silhouette_score"] = to_serializable(sil_val.get("mean"))
                row["silhouette_score_std"] = to_serializable(sil_val.get("std"))
            else:
                row["silhouette_score"] = to_serializable(sil_val)

        else:
            # Somente tempo de embed (questions only)
            val = metrics.get("embedding_time_sec")
            row["embedding_time_sec"] = to_serializable(val)

        flat_results.append(row)

    # Salvar CSV
    pd.DataFrame(flat_results).to_csv(results_path_csv, index=False)
    logging.info(f"✅ Resultados salvos em {results_path_json} e {results_path_csv}")

    # Inserir no MongoDB
    test_results_col = db["test_results"]
    timestamp_now = datetime.now(timezone.utc)

    for row in flat_results:
        doc = {
            "timestamp": timestamp_now,
            "model": row["model"],
            "log_source": f"embedding_evaluation_{dataset_source}",
            **row
        }
        test_results_col.insert_one(doc)

    logging.info(f"📊 Resultados inseridos no MongoDB em 'test_results' "
                 f"(log_source=embedding_evaluation_{dataset_source})")


# ==== Main ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de embeddings com datasets diferentes.")
    parser.add_argument(
        "--dataset", 
        choices=["hf", "local", "questions"], 
        required=True,
        help="hf = Hugging Face, local = CSV local, questions = lista de perguntas")
    args = parser.parse_args()

    if args.dataset == "hf":
        df, TEXT_COLUMN, LABEL_COLUMN = load_hf_dataset()
        evaluate_with_labels(df, TEXT_COLUMN, LABEL_COLUMN, n_clusters=8, dataset_source="hf")

    elif args.dataset == "local":
        df, TEXT_COLUMN, LABEL_COLUMN = load_local_dataset()
        evaluate_with_labels(df, TEXT_COLUMN, LABEL_COLUMN, n_clusters=3, dataset_source="local")

    elif args.dataset == "questions":
        questions = load_questions_dataset()
        evaluate_questions_only(questions)
