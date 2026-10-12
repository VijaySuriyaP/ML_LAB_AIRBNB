# Predicting Airbnb Listing Price Across Different Cities

UE24CS352A – Machine Learning mini-project.

We predict the nightly price of Airbnb listings in **New York City, Paris and Berlin** from
listing features (size, amenities, location, reviews, free text, dates) and study whether a
neural network trained on several cities **transfers** to an unseen city.
The implementation follows the approach of *"Predicting Airbnb Listing Price Across Different
Cities"* (Luo, Zhou, Zhou – 2019).

> **Team:** _<name 1 – SRN>_, _<name 2 – SRN>_  |  **Problem #:** _<your assigned number>_  |  **Section:** _<section>_

## What is implemented

| Step | Where | Details |
|------|-------|---------|
| Data cleaning | `src/data_loader.py` | `$1,234.00` strings → numbers, drop price ≤ 0, **price cap $500** (~1 % outliers), 7:2:1 train/val/test split |
| Label transform | `src/data_loader.py` | `log`, `sqrt` or none (log works best, as in the paper) |
| Continuous features (C) | `src/features.py` | ~28 numeric columns, fee columns filled with 0, others with the mean, highly-correlated (>0.9) columns dropped, standardised, optional 2-degree interaction terms |
| Categorical features (O) | `src/features.py` | One-hot encoding, plus multi-hot dictionary encoding for list columns (`amenities`, `host_verifications`) |
| Text features (T) | `src/features.py` | TF-IDF on unigrams + bigrams → truncated SVD to 50 dims per text column |
| Date features (D) | `src/features.py` | Days since earliest date; nulls filled with the mean date |
| Models | `src/models.py` | Ridge (L2) linear regression, KNN, random forest, XGBoost, MLP neural network |
| Metrics | `src/evaluate.py` | R² (primary) and MSE, plus MAE in dollars; predicted-vs-actual scatter plots |
| Cross-city learning | `src/transfer.py` | Train on NYC / Paris / NYC+Paris, evaluate on each city's test set and on unseen **Berlin** |
| Demo | `demo.py` | Loads a trained model and predicts prices for held-out listings |
| Tests | `tests/` | Pytest smoke tests on synthetic data |

The feature builder is fit on **training data only** and then applied to validation/test/unseen
cities, so there is no leakage. Columns missing in a city (e.g. a city-specific neighbourhood
field) are treated as missing values, which allows one model to be trained on several cities.

## Project layout

```
airbnb-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── demo.py                  # live demo entry point
├── data/
│   ├── README.md            # how to download the Kaggle data
│   └── sample/              # synthetic CSVs (generated, optional)
├── src/
│   ├── config.py            # column lists, paths, defaults
│   ├── data_loader.py       # load / clean / split / label transforms
│   ├── features.py          # FeatureBuilder (C, O, T, D)
│   ├── models.py            # model definitions
│   ├── evaluate.py          # metrics and plots
│   ├── train.py             # single-city experiments
│   ├── transfer.py          # cross-city + unseen-city experiments
│   └── synthetic.py         # synthetic data generator
├── tests/test_pipeline.py
├── models/                  # saved models (git-ignored)
└── results/                 # CSV results and plots (git-ignored)
```

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Download the three Kaggle datasets and save them as `data/nyc.csv`, `data/paris.csv`,
`data/berlin.csv` – details in [`data/README.md`](data/README.md).
To try the code without the real data: `python -m src.synthetic` (writes `data/sample/`),
then add `--data-dir data/sample` to any command below.

## Run

```bash
# 1. Quick sanity check
pytest -q

# 2. Single-city experiments (writes results/<city>_results.csv, scatter plots, models/*.joblib)
python -m src.train --city nyc --models linear knn rf xgb nn --features C --label none     # raw price, continuous only
python -m src.train --city nyc --models xgb --features C --label log                       # log label
python -m src.train --city nyc --models xgb --features C --label log --interactions        # + interaction terms
python -m src.train --city nyc --models xgb --features C+O+T --label log                   # all features
python -m src.train --city paris --models xgb --features C+O+T --label log

# 3. Neural-network transfer learning (NYC / Paris / NYC+Paris -> Berlin)
python -m src.transfer --features C+O --label log

# 4. Live demo
python demo.py --city nyc --model xgb --n 8
```

Useful flags for `src.train`: `--features` (any of `C O T D` joined with `+`), `--label {log,sqrt,none}`,
`--cap` (price threshold, `0` keeps all listings), `--interactions`, `--data-dir`, `--path`,
`--models` (`linear knn rf xgb nn nn_adam`).

## Models

* **linear** – Ridge regression (L2) · **knn** – unweighted KNN, Euclidean distance (k = 5)
* **rf** – random forest, 100 trees · **xgb** – gradient-boosted trees
* **nn** – MLP (64, 32), ReLU, SGD + Nesterov momentum (μ = 0.9, lr = 0.005), L2 = 0.005,
  early stopping on 10 % of the training data (stop when validation R² improves < 1e-4 for 10 iterations)
* **nn_adam** – the larger first network (128, 512, 64), Adam (lr = 1e-4, weight decay = 1e-5), batch size 64

## Results

Fill this in from your own runs (`results/*.csv`) – the paper's numbers are only a reference point:

| Experiment | Paper (test R²) | Ours |
|-----------|-----------------|------|
| XGBoost, continuous, price ≤ $500 (NYC) | 0.665 | _TBD_ |
| XGBoost, continuous, log label (NYC) | 0.669 | _TBD_ |
| XGBoost, C+O+T, log label (NYC) | 0.740 | _TBD_ |
| XGBoost, C+O+T, log label (Paris) | 0.704 | _TBD_ |
| NN trained on NYC + Paris (own test) | 0.773 | _TBD_ |
| NN trained on NYC + Paris → Berlin | 0.514 | _TBD_ |

## Notes and limitations

* Prices in the Kaggle files are treated as one currency, as in the paper.
* Column names must match the Kaggle "listings" schema; unknown columns are ignored and
  missing ones are treated as null.
* Results vary slightly with the random seed (`--seed`).

## Reference

Y. Luo, X. Zhou, Y. Zhou, *Predicting Airbnb Listing Price Across Different Cities*, 2019.
