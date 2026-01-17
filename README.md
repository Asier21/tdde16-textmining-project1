# TDDE16 – Text Mining Project

## Semantic Robustness vs Computational Cost (AG News)

This repository contains the code for an **individual** TDDE16 project comparing text representations for news classification on **AG News**, focusing on the trade-off between **semantic robustness** and **computational cost**.

### Compared representations

- TF-IDF (unigrams + bigrams)
- Averaged fastText embeddings (cc.en.300)
- Sentence embeddings (SBERT: all-MiniLM-L6-v2)

All methods use the same classifier (Logistic Regression) to isolate the effect of the representation.

---

## Repository layout

- `data/ag_news/`  
  `train.csv`, `test.csv` (AG News from Kaggle)

- `models/fasttext/`  
  `cc.en.300.bin` (not included in git due to size)

- `src/`  
  `data.py` – load/preprocess dataset  
  `representations.py` – fastText + SBERT embeddings  
  `train_eval.py` – TF-IDF baseline  
  `timing.py` – time + metrics for TF-IDF / fastText / SBERT  
  `plot_tradeoff.py` – generates trade-off figure  
  `error_analysis.py` – exports misclassification examples

- `results/`  
  outputs such as `fig_tradeoff.png` and `error_analysis_examples.csv`

- `report/`  
  LaTeX paper (`paper.tex`) + bibliography (`custom.bib`)

---

## Setup

Recommended: create a clean conda environment (example name: `tdde16_p`) and install dependencies.

Minimum dependencies:

- numpy, pandas, scipy, scikit-learn
- sentence-transformers
- fasttext
- matplotlib

---

## fastText model

Download the official English model and place it here:
`models/fasttext/cc.en.300.bin`

Example download via Python:

```

Then move cc.en.300.bin into:
models/fasttext/

```

## How to run

All commands should be executed from the root of the repository.

### TF-IDF baseline

```bash
python -m src.train_eval
```

Timing (vectorization, training, inference) for all representations
python -m src.timing

Generate trade-off figure
python src/plot_tradeoff.py

Export error analysis examples
python -m src.error_analysis

Dataset

The AG News dataset is used with its predefined train/test split.

4 classes: World, Sports, Business, Sci/Tech

Approximately 120,000 training samples

Approximately 7,600 test samples

Dataset source (Kaggle):
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

## Author

Asier Iglesias Alconero
Linköping University
