# services/embeddings.py
# ===============================
## python -m services.embeddings

import logging
from functools import lru_cache
from typing import Union, List
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# =========================
# Device configuration
# =========================
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"🔹 Using device: {DEFAULT_DEVICE}")

# =========================
# Auxiliary function: truncation warning
# =========================
def check_truncation(tokenizer, texts: List[str], max_length: int):
    for t in texts:
        if len(tokenizer.encode(t, truncation=False)) > max_length:
            logging.warning(f"⚠ Truncated text for {max_length} tokens: {t[:50]}...")


# =========================
# Lazy Load - HuggingFace BERT
# =========================
@lru_cache(maxsize=1)
def load_bert(device=DEFAULT_DEVICE):
    logging.info("🔹 Loading model BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model


def get_bert_embeddings(texts: Union[str, List[str]], device=DEFAULT_DEVICE) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    tokenizer, model = load_bert()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# =========================
# Lazy Load - HuggingFace GPT-2
# =========================
@lru_cache(maxsize=1)
def load_gpt2():
    logging.info("🔹 Loading model GPT-2 (embeddings de hidden states)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def get_gpt2_embeddings(texts: Union[str, List[str]], device=DEFAULT_DEVICE) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    tokenizer, model = load_gpt2()
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# =========================
# Lazy Load - Sentence-BERT (SBERT)
# =========================
@lru_cache(maxsize=1)
def load_sbert():
    logging.info("🔹 Loading model SBERT...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_sbert_embeddings(texts: Union[str, List[str]], device=DEFAULT_DEVICE) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    model = load_sbert()
    return np.array(model.encode(texts, convert_to_numpy=True))


# =========================
# Embedding benchmark utility
# =========================
def benchmark_embedding(model_name: str, embed_fn, texts: List[str], **kwargs):
    import time
    start = time.time()
    emb = embed_fn(texts, **kwargs) 
    elapsed = time.time() - start
    logging.info(f"⏱ {model_name} - Time: {elapsed:.3f}s - Shape: {emb.shape}")
    return emb, elapsed

def benchmark_all_models(texts: List[str], device=DEFAULT_DEVICE):
    logging.info("📊 Running benchmark of all models...")
    results = {}
    for name, fn in [
        ("BERT", get_bert_embeddings),
        ("GPT-2", get_gpt2_embeddings),
        ("SBERT", get_sbert_embeddings),
    ]:
        _, elapsed = benchmark_embedding(name, fn, texts, device=device)
        results[name] = elapsed

    # Comparison chart
    plt.bar(results.keys(), results.values(), color=["blue", "green", "orange"])
    plt.ylabel("Tempo (segundos)")
    plt.title("Benchmark de embeddings")
    plt.show()
    return results

# =========================
# Execução local para teste
# =========================
if __name__ == "__main__":
    logging.info("Rodando testes locais no embeddings.py...")
    sample_text = ["How can I reset my password?"]
    benchmark_all_models(sample_text)