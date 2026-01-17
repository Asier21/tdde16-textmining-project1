from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import load_ag_news

def run_tfidf_baseline():
    X_train, y_train, X_test, y_test, _, _ = load_ag_news()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        solver="lbfgs"
    )
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("TF-IDF + Logistic Regression")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {f1:.4f}")
    print("\nPer-class report:\n")
    print(classification_report(y_test, preds, digits=4))

if __name__ == "__main__":
    run_tfidf_baseline()
