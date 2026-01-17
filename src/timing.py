import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_ag_news
from src.representations import load_fasttext_model, fasttext_avg_embeddings
from src.representations import load_sbert_model, sbert_embeddings


def median_time(fn, repeats=3):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times)), out


def time_tfidf_pipeline():
    X_train, y_train, X_test, y_test, _, _ = load_ag_news()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
    )

    def vec_step():
        Xtr = vectorizer.fit_transform(X_train)
        Xte = vectorizer.transform(X_test)
        return Xtr, Xte

    vec_time, (Xtr, Xte) = median_time(vec_step, repeats=3)

    clf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)

    def train_step():
        clf.fit(Xtr, y_train)
        return clf

    train_time, _ = median_time(train_step, repeats=3)

    def infer_step():
        return clf.predict(Xte)

    infer_time, preds = median_time(infer_step, repeats=3)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("TF-IDF timing (median of 3 runs)")
    print(f"Vectorization time: {vec_time:.3f} s")
    print(f"Training time:      {train_time:.3f} s")
    print(f"Inference time:     {infer_time:.3f} s")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro-F1:           {f1:.4f}")
    print(f"Vector dim:         {len(vectorizer.get_feature_names_out())}")


def time_fasttext_pipeline():
    X_train, y_train, X_test, y_test, _, _ = load_ag_news()

    ft = load_fasttext_model("models/fasttext/cc.en.300.bin")

    def vec_step():
        Xtr = fasttext_avg_embeddings(X_train, ft)
        Xte = fasttext_avg_embeddings(X_test, ft)
        return Xtr, Xte

    vec_time, (Xtr, Xte) = median_time(vec_step, repeats=3)

    clf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)

    def train_step():
        clf.fit(Xtr, y_train)
        return clf

    train_time, _ = median_time(train_step, repeats=3)

    def infer_step():
        return clf.predict(Xte)

    infer_time, preds = median_time(infer_step, repeats=3)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("fastText (avg) timing (median of 3 runs)")
    print(f"Vectorization time: {vec_time:.3f} s")
    print(f"Training time:      {train_time:.3f} s")
    print(f"Inference time:     {infer_time:.3f} s")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro-F1:           {f1:.4f}")
    print(f"Vector dim:         {Xtr.shape[1]}")

def time_sbert_pipeline():
    X_train, y_train, X_test, y_test, _, _ = load_ag_news()

    sbert = load_sbert_model("all-MiniLM-L6-v2")

    def vec_step():
        Xtr = sbert_embeddings(X_train, sbert, batch_size=64)
        Xte = sbert_embeddings(X_test, sbert, batch_size=64)
        return Xtr, Xte

    vec_time, (Xtr, Xte) = median_time(vec_step, repeats=1)
    # Nota: SBERT es caro -> repeats=1 para no tardar siglos. (Luego si quieres hacemos 3 en test.)

    clf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)

    def train_step():
        clf.fit(Xtr, y_train)
        return clf

    train_time, _ = median_time(train_step, repeats=3)

    def infer_step():
        return clf.predict(Xte)

    infer_time, preds = median_time(infer_step, repeats=3)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("SBERT timing (MiniLM-L6-v2)")
    print(f"Vectorization time: {vec_time:.3f} s")
    print(f"Training time:      {train_time:.3f} s")
    print(f"Inference time:     {infer_time:.3f} s")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Macro-F1:           {f1:.4f}")
    print(f"Vector dim:         {Xtr.shape[1]}")

if __name__ == "__main__":
    time_tfidf_pipeline()
    print()
    time_fasttext_pipeline()
    print()
    time_sbert_pipeline()
