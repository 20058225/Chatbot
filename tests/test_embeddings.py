# tests/test_embeddings.py
# =======================
## 
import os
import json
import torch
import numpy as np
import pandas as pd
import pytest
import mongomock
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import services.embeddings as emb
import services.evaluation as eval_mod
import simulation.evaluate_embedding as sim_eval

@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    client = mongomock.MongoClient()
    db = client["test_db"]
    monkeypatch.setattr(sim_eval, "db", db)
    monkeypatch.setattr(eval_mod, "db", db)
    yield db

# ==== services/embeddings.py ====
def test_check_truncation_warns(caplog):
    tok_mock = MagicMock()
    tok_mock.encode.side_effect = lambda t, truncation=False: list(range(200))
    emb.check_truncation(tok_mock, ["x"*100], max_length=10)
    assert "⚠ Texto truncado" in caplog.text

@patch("services.embeddings.load_bert")
def test_get_bert_embeddings_returns_array(mock_load):
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.zeros((1,5), dtype=torch.long)}
    model = MagicMock()
    model.return_value = MagicMock(last_hidden_state=torch.zeros((1, 5, 8)))
    mock_load.return_value = (tok, model)

    arr = emb.get_bert_embeddings("hello")
    assert isinstance(arr, np.ndarray)

@patch("services.embeddings.get_bert_embeddings", return_value=np.zeros((1,8)))
@patch("services.embeddings.get_gpt2_embeddings", return_value=np.zeros((1,8)))
@patch("services.embeddings.get_sbert_embeddings", return_value=np.zeros((1,8)))
@patch("services.embeddings.plt")
def test_benchmark_all_models_runs(mock_plt, *_):
    res = emb.benchmark_all_models(["test"])
    assert all(m in res for m in ["BERT","GPT-2","SBERT"])

# ==== services/evaluation.py ====
def test_evaluate_classification_and_clustering():
    def dummy_embed(texts):
        embeddings = []
        for t in texts:
            if t in ["a", "b"]:
                embeddings.append([0.1, 0.2])
            else:
                embeddings.append([1.0, 1.0])
        return np.array(embeddings)
    
    X_train, y_train = ["a", "b"], [0, 1]
    X_test, y_test = ["c", "d"], [0, 1]

    cls = eval_mod.evaluate_classification(dummy_embed, X_train, y_train, X_test, y_test)
    clu = eval_mod.evaluate_clustering(dummy_embed, ["a", "b", "c"], n_clusters=2)
    
    assert "accuracy" in cls and "silhouette_score" in clu

def dummy_embed(texts):
    embeddings = []
    for t in texts:
        if t in ["a", "b"]:
            embeddings.append([0.1, 0.2])
        else:
            embeddings.append([1.0, 1.0])
    return np.array(embeddings)

@patch("services.evaluation.plot_results")
def test_run_full_evaluation_returns_expected(mock_plot):
    dummy_embed_fn = dummy_embed
    dataset = {
        "X_train": ["a", "b"], "y_train": [0, 1],
        "X_test": ["c", "d"], "y_test": [0, 1],
        "texts": ["a", "b", "c", "d"]
    }
    embed_fns = {"DUMMY": dummy_embed_fn}
    res = eval_mod.run_full_evaluation(embed_fns, dataset, repeats=2, n_clusters=2)
    assert "classification" in res["DUMMY"]

# ==== simulation/evaluate_embedding.py ====
@patch("simulation.evaluate_embedding.run_full_evaluation", return_value={
    "BERT": {
        "classification": {
            "accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1_score": 0.6,
            "embedding_train_time_sec": 1.0, "embedding_test_time_sec": 1.1
        },
        "clustering": {"silhouette_score": 0.5}
    }
})
def test_evaluate_with_labels_saves_results(mock_run, tmp_path, monkeypatch):
    df = pd.DataFrame({
        "text": ["a", "a2", "b", "b2"],
        "label": [0, 0, 1, 1]
    })
    monkeypatch.chdir(tmp_path)
    sim_eval.evaluate_with_labels(df, "text", "label", 2, "testsrc")
    assert os.path.exists("results/testsrc.json")



@patch("services.embeddings.get_bert_embeddings", return_value=np.zeros((1,8)))
@patch("services.embeddings.get_gpt2_embeddings", return_value=np.zeros((1,8)))
@patch("services.embeddings.get_sbert_embeddings", return_value=np.zeros((1,8)))
def test_evaluate_questions_only_saves(_, __, ___, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim_eval.evaluate_questions_only(["q1","q2"])
    files = os.listdir("results")
    assert any(f.endswith(".json") for f in files)

def test_save_results_creates_files_and_mongo(tmp_path, mock_mongo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = {"BERT":{"classification":{"accuracy":0.9,"precision":0.8,"recall":0.7,"f1_score":0.6,"embedding_train_time_sec":1.0,"embedding_test_time_sec":1.1},"clustering":{"silhouette_score":0.5}}}
    sim_eval.save_results(res, "unit_test_src")
    assert os.path.exists("results/unit_test_src.json")
    assert mock_mongo["test_results"].count_documents({}) > 0
