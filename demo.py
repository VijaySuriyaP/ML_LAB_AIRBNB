"""Live demo: load a trained model and predict nightly prices for held-out listings.

    python -m src.train --city nyc --models xgb --features C+O+T      # once, to train + save
    python demo.py --city nyc --model xgb --n 8                        # predict 8 random test listings

If the saved model does not exist yet, the demo trains a quick XGBoost model first.
Use --data-dir data/sample to run on the synthetic data.
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src import config
from src.data_loader import inverse_label, prepare_city
from src.train import run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", default="nyc", choices=config.CITIES)
    p.add_argument("--model", default="xgb")
    p.add_argument("--n", type=int, default=8, help="number of test listings to show")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--features", default="C+O+T")
    p.add_argument("--seed", type=int, default=config.SEED)
    a = p.parse_args(argv)

    saved = config.MODELS_DIR / f"{a.city}_{a.model}.joblib"
    if not saved.exists():
        print(f"No saved model at {saved} - training a quick one first...\n")
        run(a.city, [a.model], a.features, "log", config.PRICE_CAP, data_dir=a.data_dir, plots=False)
        print()
    bundle = joblib.load(saved)
    model, fb, label = bundle["model"], bundle["builder"], bundle["label"]

    _, _, test = prepare_city(a.city, a.data_dir)
    sample = test.sample(min(a.n, len(test)), random_state=a.seed).reset_index(drop=True)
    pred = inverse_label(model.predict(fb.transform(sample)), label)

    cols = [c for c in ("neighbourhood_cleansed", "room_type", "accommodates", "bedrooms") if c in sample.columns]
    out = sample[cols].copy()
    out["actual_price"] = sample["price"].round(0)
    out["predicted_price"] = pred.round(0)
    out["abs_error"] = (out["predicted_price"] - out["actual_price"]).abs()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(out.to_string(index=False))
    print(f"\nMean absolute error on this sample: {out['abs_error'].mean():.1f} per night")


if __name__ == "__main__":
    main()
