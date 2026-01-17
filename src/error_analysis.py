import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_ag_news
from src.representations import (
    load_fasttext_model,
    fasttext_avg_embeddings,
    load_sbert_model,
    sbert_embeddings,
)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def run_error_analysis(n_examples=5):
    X_train, y_train, X_test, y_test, _, _ = load_ag_news()

    # ---------- TF-IDF ----------
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
    )
    Xtr_tfidf = tfidf.fit_transform(X_train)
    Xte_tfidf = tfidf.transform(X_test)

    clf_tfidf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
    clf_tfidf.fit(Xtr_tfidf, y_train)
    preds_tfidf = clf_tfidf.predict(Xte_tfidf)

    # ---------- SBERT ----------
    sbert = load_sbert_model("all-MiniLM-L6-v2")
    Xtr_sbert = sbert_embeddings(X_train, sbert, batch_size=64)
    Xte_sbert = sbert_embeddings(X_test, sbert, batch_size=64)

    clf_sbert = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
    clf_sbert.fit(Xtr_sbert, y_train)
    preds_sbert = clf_sbert.predict(Xte_sbert)

    rows = []

    for i, text in enumerate(X_test):
        true = y_test[i]
        tf_ok = preds_tfidf[i] == true
        sb_ok = preds_sbert[i] == true

        if (not tf_ok and sb_ok) or (tf_ok and not sb_ok):
            rows.append({
                "text": text,
                "true_label": LABELS[true],
                "tfidf_pred": LABELS[preds_tfidf[i]],
                "sbert_pred": LABELS[preds_sbert[i]],
                "case": "TF-IDF wrong, SBERT correct" if (not tf_ok and sb_ok)
                        else "TF-IDF correct, SBERT wrong"
            })

    df = pd.DataFrame(rows)

    # Tomamos ejemplos equilibrados
    examples = (
        df.groupby("case")
        .head(n_examples)
        .reset_index(drop=True)
    )

    examples.to_csv("results/error_analysis_examples.csv", index=False)
    print("Saved examples to results/error_analysis_examples.csv")
    print(examples.head())


if __name__ == "__main__":
    run_error_analysis(n_examples=3)
