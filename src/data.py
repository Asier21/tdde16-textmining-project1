import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/ag_news")

def load_ag_news():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    # Kaggle AG News: 3 columnas (Class Index, Title, Description) y primera fila = header dentro de los datos
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # Eliminar la primera fila que contiene strings (header)
    train_df = train_df.iloc[1:].reset_index(drop=True)
    test_df = test_df.iloc[1:].reset_index(drop=True)

    # Nombrar columnas
    train_df.columns = ["label", "title", "description"]
    test_df.columns = ["label", "title", "description"]

    # Convertir labels a int y pasar de 1-4 a 0-3 (más cómodo)
    train_df["label"] = train_df["label"].astype(int) - 1
    test_df["label"] = test_df["label"].astype(int) - 1

    # Texto final: title + description
    train_df["text"] = (train_df["title"].astype(str) + ". " + train_df["description"].astype(str)).str.strip()
    test_df["text"] = (test_df["title"].astype(str) + ". " + test_df["description"].astype(str)).str.strip()

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    return X_train, y_train, X_test, y_test, train_df, test_df

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, train_df, test_df = load_ag_news()
    print("Train:", len(X_train), "Test:", len(X_test))
    print("Labels (train) min/max:", min(y_train), max(y_train))
    print("Example text:\n", X_train[0][:300])
