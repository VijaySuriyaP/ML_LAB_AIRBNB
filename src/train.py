"""Train and evaluate models on a single city.

Examples
--------
    python -m src.train --city nyc --models linear knn rf xgb nn --features C+O+T --label log
    python -m src.train --city paris --models xgb --features C --label none --cap 0
    python -m src.train --city nyc --models xgb --features C --interactions
    python -m src.train --city nyc --data-dir data/sample          # synthetic data
"""
from __future__ import annotations

import argparse
import time

import joblib
import pandas as pd

from . import config
from .data_loader import inverse_label, prepare_city, transform_label
from .evaluate import regression_metrics, scatter_plot
from .features import FeatureBuilder
from .models import MODEL_NAMES, get_model


def run(city, models, features, label, cap, interactions=False, data_dir=None, path=None,
        save=True, plots=True, seed=config.SEED):
    cap = None if (cap is None or cap <= 0) else cap
    train, val, test = prepare_city(city, data_dir, path, cap, seed)
    print(f"[{city}] listings  train={len(train)}  val={len(val)}  test={len(test)}")

    fb = FeatureBuilder(features, interactions=interactions, seed=seed)
    X_train = fb.fit_transform(train)
    X_val, X_test = fb.transform(val), fb.transform(test)
    y_train = transform_label(train["price"], label)
    y_val = transform_label(val["price"], label)
    y_test = transform_label(test["price"], label)
    print(f"[{city}] feature matrix: {X_train.shape[1]} columns ({features}"
          f"{', +interactions' if interactions else ''})")

    rows = []
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    for name in models:
        t0 = time.time()
        model = get_model(name, seed)
        model.fit(X_train, y_train)
        row = {"city": city, "model": name, "features": features + ("+I" if interactions else ""),
               "label": label, "cap": cap or "none"}
        for split_name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
            m = regression_metrics(y, model.predict(X))
            row[f"{split_name}_mse"], row[f"{split_name}_r2"] = m["mse"], m["r2"]
        # also report error in dollars on the test set
        pred_price = inverse_label(model.predict(X_test), label)
        row["test_mae_usd"] = float(abs(pred_price - test["price"].to_numpy()).mean())
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  {name:8s} train R2={row['train_r2']:.3f}  val R2={row['val_r2']:.3f}  "
              f"test R2={row['test_r2']:.3f}  test MSE={row['test_mse']:.4f}  ({row['seconds']}s)")
        if plots:
            scatter_plot(y_test, model.predict(X_test), config.RESULTS_DIR / f"{city}_{name}_scatter.png",
                         f"{city.upper()} - {name} (test)")
        if save:
            joblib.dump({"model": model, "builder": fb, "label": label, "features": features,
                         "city": city}, config.MODELS_DIR / f"{city}_{name}.joblib")

    results = pd.DataFrame(rows)
    results.to_csv(config.RESULTS_DIR / f"{city}_results.csv", index=False)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", required=True, choices=config.CITIES)
    p.add_argument("--models", nargs="+", default=["linear", "knn", "rf", "xgb", "nn"], choices=MODEL_NAMES)
    p.add_argument("--features", default="C+O+T", help="any of C (continuous) O (categorical) T (text) D (date), joined by +")
    p.add_argument("--label", default="log", choices=["log", "sqrt", "none"])
    p.add_argument("--cap", type=float, default=config.PRICE_CAP, help="drop listings above this price; 0 = keep all")
    p.add_argument("--interactions", action="store_true", help="add 2-degree interaction terms of continuous features")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--path", default=None, help="explicit CSV path (overrides --data-dir)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    a = p.parse_args(argv)
    res = run(a.city, a.models, a.features, a.label, a.cap, a.interactions, a.data_dir, a.path,
              save=not a.no_save, plots=not a.no_plots, seed=a.seed)
    print("\nSaved results to", config.RESULTS_DIR / f"{a.city}_results.csv")
    return res


if __name__ == "__main__":
    main()
