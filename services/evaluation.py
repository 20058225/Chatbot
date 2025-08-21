# services/evaluation.py
# ===============================
## python -m services.evaluation

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.cluster import KMeans
from typing import List, Callable, Tuple, Dict, Any
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from services.mongo import db

# Importa as funções de embeddings já otimizadas
from services.embeddings import get_bert_embeddings, get_gpt2_embeddings, get_sbert_embeddings

logging.basicConfig(level=logging.INFO)

# =========================
# Classificação
# =========================
def evaluate_classification(
    embed_fn: Callable[[List[str]], np.ndarray],
    X_train: List[str], y_train: List,
    X_test: List[str], y_test: List,
    classifier=None, device: str = "cpu"
) -> dict:
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)

    # 🔹 Garantir que labels são strings
    y_train = [str(y).strip() for y in y_train]
    y_test = [str(y).strip() for y in y_test]

    # 🔹 Remover entradas com label vazio
    clean_data = [(x, y) for x, y in zip(X_train, y_train) if y != "" and y.lower() != "unknown"]
    X_train, y_train = zip(*clean_data)

    clean_data = [(x, y) for x, y in zip(X_test, y_test) if y != "" and y.lower() != "unknown"]
    X_test, y_test = zip(*clean_data)

    logging.info("Generating train embeddings...")
    start = time.time()
    X_train_emb = embed_fn(X_train)
    train_time = time.time() - start

    logging.info("Generating test embeddings...")
    start = time.time()
    X_test_emb = embed_fn(X_test)
    test_time = time.time() - start

    logging.info("Training classifier...")
    classifier.fit(X_train_emb, y_train)

    logging.info("Predicting on test set...")
    y_pred = classifier.predict(X_test_emb)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "embedding_train_time_sec": train_time,
        "embedding_test_time_sec": test_time
    }
    return metrics

# =========================
# Clustering
# =========================
def evaluate_clustering(
    embed_fn: Callable[..., np.ndarray],
    texts: List[str],
    n_clusters: int = 8,
    sample_size: int = None,
    device: str = "cpu"
) -> dict:
    logging.info("Generating embeddings for clustering...")
    start = time.time()
    embeddings = embed_fn(texts)
    embed_time = time.time() - start

    logging.info("🔹 Padronizando embeddings para K-Means...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    logging.info("🔹 Calculando Silhouette Score...")
    silhouette = silhouette_score(embeddings, labels)

    return {
        "silhouette_score": silhouette,
        "embedding_time_sec": embed_time
    }

# =========================
# Avaliação completa
# =========================
def run_full_evaluation(
    embed_fns: Dict[str, Callable[..., np.ndarray]],
    dataset: dict,
    classifier=None,
    n_clusters: int = 8,
    repeats: int = 1,
    sample_size: int = None,
    device: str = "cpu"
) -> dict:
    results = {}

    for name, embed_fn in embed_fns.items():
        logging.info(f"Evaluating embeddings from {name}...")
        cls_scores = []
        clu_scores = []

        for r in range(repeats):
            logging.info(f"--- Execução {r+1}/{repeats} ---")
            cls_metrics = evaluate_classification(
                embed_fn,
                dataset["X_train"], dataset["y_train"],
                dataset["X_test"], dataset["y_test"],
                classifier=classifier,
                device=device
            )
            cluster_metrics = evaluate_clustering(
                embed_fn,
                dataset["texts"],
                n_clusters=n_clusters,
                sample_size=sample_size,
                device=device
            )
            cls_scores.append(cls_metrics)
            clu_scores.append(cluster_metrics)

        def agg_metric(scores_list, key):
            vals = [s[key] for s in scores_list]
            return {"mean": np.mean(vals), "std": np.std(vals)}

        results[name] = {
            "classification": {k: agg_metric(cls_scores, k) for k in cls_scores[0].keys()},
            "clustering": {k: agg_metric(clu_scores, k) for k in clu_scores[0].keys()}
        }

    plot_results(results)
    return results

# =========================
# Visualização
# =========================
def plot_results(results: dict):
    models = list(results.keys())

    # Classificação - Accuracy
    acc_means = [results[m]["classification"]["accuracy"]["mean"] for m in models]
    acc_stds = [results[m]["classification"]["accuracy"]["std"] for m in models]

    plt.figure(figsize=(8, 4))
    plt.bar(models, acc_means, yerr=acc_stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title("Comparação de Acurácia")
    plt.show()

    # Clustering - Silhouette
    sil_means = [results[m]["clustering"]["silhouette_score"]["mean"] for m in models]
    sil_stds = [results[m]["clustering"]["silhouette_score"]["std"] for m in models]

    plt.figure(figsize=(8, 4))
    plt.bar(models, sil_means, yerr=sil_stds, capsize=5, color="orange")
    plt.ylabel("Silhouette Score")
    plt.title("Comparação de Clustering")
    plt.show()


# =========================
# Execução local
# =========================
if __name__ == "__main__":
    logging.info("Rodando avaliação de teste...")

    # Dataset fictício
    dataset = {
        "X_train": ["ticket issue A", "ticket issue B", "ticket issue C"],
        "y_train": [0, 1, 0],
        "X_test": ["ticket issue D", "ticket issue E"],
        "y_test": [1, 0],
        "texts": ["ticket issue A", "ticket issue B", "ticket issue C", "ticket issue D", "ticket issue E"]
    }

    embed_fns = {
        "BERT": get_bert_embeddings,
        "GPT-2": get_gpt2_embeddings,
        "SBERT": get_sbert_embeddings
    }

    run_full_evaluation(embed_fns, dataset, repeats=2, n_clusters=2, device="cpu", sample_size=5)
