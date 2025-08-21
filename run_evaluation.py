import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, davies_bouldin_score, f1_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from services.embeddings import get_bert_embeddings, get_gpt2_embeddings

RESULTS_DIR = "simulation/result"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "bert": get_bert_embeddings,
    "gpt2": get_gpt2_embeddings
}

def load_datasets():
    df = pd.read_csv("data/train_model.csv")
    if "text" not in df.columns and "description" in df.columns:
        df = df.rename(columns={"description": "text"})
    return df.dropna(subset=['text', 'sentiment', 'priority'])


def generate_embeddings(df, model_name):
    print(f"[INFO] Generating embeddings with {model_name.upper()}...")
    return MODELS[model_name](df['text'].tolist())

def run_classification(X, y, model_name, target_name):
    smote = SMOTE()
    X_bal, y_bal = smote.fit_resample(X, y)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_bal, y_bal)
    preds = clf.predict(X)

    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'{target_name} - {model_name}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f'{RESULTS_DIR}/confusion_{target_name.lower()}_{model_name}.png')
    plt.close()
    return report

def run_clustering(X, model_name):
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    silhouette = silhouette_score(X_pca, cluster_labels)
    db_index = davies_bouldin_score(X_pca, cluster_labels)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cluster_labels, palette='tab10', s=50)
    plt.title(f't-SNE Clusters - {model_name.upper()}')
    plt.savefig(f'{RESULTS_DIR}/tsne_clusters_{model_name}.png')
    plt.close()

    return silhouette, db_index

def measure_latency(df, model_name):
    times = []
    for text in df['text'].tolist():
        start = time.perf_counter()
        _ = MODELS[model_name]([text])
        end = time.perf_counter()
        times.append((end - start) * 1000)

    plt.figure(figsize=(8, 6))
    sns.histplot(times, bins=20, kde=True)
    plt.title(f'Latency - {model_name.upper()}')
    plt.xlabel('Milliseconds')
    plt.ylabel('Frequency')
    plt.savefig(f'{RESULTS_DIR}/latency_{model_name}.png')
    plt.close()

    return np.median(times)

def run_macro_f1(X, y_sentiment, y_priority, model_name):
    clf_sent = LogisticRegression(max_iter=1000).fit(X, y_sentiment)
    clf_prio = LogisticRegression(max_iter=1000).fit(X, y_priority)
    macro_f1_sent = f1_score(y_sentiment, clf_sent.predict(X), average='macro')
    macro_f1_prio = f1_score(y_priority, clf_prio.predict(X), average='macro')

    plt.figure(figsize=(6, 4))
    plt.bar(["Sentiment", "Priority"], [macro_f1_sent, macro_f1_prio], color=['blue', 'green'])
    plt.ylabel('Macro-F1 Score')
    plt.title(f'Macro-F1 Comparison - {model_name.upper()}')
    plt.savefig(f'{RESULTS_DIR}/f1_macro_comparison_{model_name}.png')
    plt.close()

    return macro_f1_sent, macro_f1_prio

if __name__ == "__main__":
    df = load_datasets()

    for model_name in MODELS.keys():
        print(f"\n===== Running Evaluation for {model_name.upper()} =====")
        X = generate_embeddings(df, model_name)

        sentiment_report = run_classification(X, df['sentiment'], model_name, "Sentiment")
        priority_report = run_classification(X, df['priority'], model_name, "Priority")

        silhouette, db_index = run_clustering(X, model_name)
        latency = measure_latency(df, model_name)
        macro_f1_sent, macro_f1_prio = run_macro_f1(X, df['sentiment'], df['priority'], model_name)

        print(f"\n--- {model_name.upper()} RESULTS ---")
        print(f"Sentiment Macro-F1: {macro_f1_sent:.4f}")
        print(f"Priority Macro-F1: {macro_f1_prio:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies–Bouldin Index: {db_index:.4f}")
        print(f"Median Latency: {latency:.2f} ms")
        print("------------------------------")
