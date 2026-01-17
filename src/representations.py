import re
import numpy as np
import fasttext

TOKEN_RE = re.compile(r"[a-z]+")

def simple_tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def load_fasttext_model(path="models/fasttext/cc.en.300.bin"):
    return fasttext.load_model(path)

def fasttext_avg_embeddings(texts, ft_model):
    dim = ft_model.get_dimension()
    X = np.zeros((len(texts), dim), dtype=np.float32)

    for i, text in enumerate(texts):
        toks = simple_tokenize(text)
        if not toks:
            continue
        vecs = np.stack([ft_model.get_word_vector(t) for t in toks], axis=0)
        X[i] = vecs.mean(axis=0)

    return X

from sentence_transformers import SentenceTransformer

def load_sbert_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def sbert_embeddings(texts, sbert_model, batch_size=64):
    # convert_to_numpy=True nos da np.ndarray directamente
    return sbert_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
