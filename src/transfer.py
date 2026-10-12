"""Cross-city experiments with the neural network (paper section 6.2).

1. Individual vs combined training (Table 6): train on NYC, Paris and NYC+Paris,
   evaluate each on its own held-out test split(s).
2. Transfer to an unseen city (Table 7): train on NYC / Paris / NYC+Paris and test on
   *all* Berlin listings.

The feature builder is fit on the training listings only. Columns that a city does not
have (e.g. a city-specific neighbourhood field) are simply missing for that city.

    python -m src.transfer
    python -m src.transfer --data-dir data/sample --features C+O
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .data_loader import clean_prices, load_city, split, transform_label
from .evaluate import regression_metrics, scatter_plot
from .features import FeatureBuilder
from .models import get_model


def _fit_eval(train_frames, eval_sets, features, label, model_name, seed):
    train = pd.concat(train_frames, ignore_index=True)
    fb = FeatureBuilder(features, seed=seed)
    X = fb.fit_transform(train)
    y = transform_label(train["price"], label)
    model = get_model(model_name, seed).fit(X, y)
    out = {"train": regression_metrics(y, model.predict(X))}
    preds = {}
    for name, df in eval_sets.items():
        yt = transform_label(df["price"], label)
        yp = model.predict(fb.transform(df))
        out[name] = regression_metrics(yt, yp)
        preds[name] = (yt, yp)
    return out, preds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-cities", nargs="+", default=["nyc", "paris"])
    p.add_argument("--target-city", default="berlin")
    p.add_argument("--features", default="C+O")
    p.add_argument("--label", default="log", choices=["log", "sqrt", "none"])
    p.add_argument("--model", default="nn")
    p.add_argument("--cap", type=float, default=config.PRICE_CAP)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--seed", type=int, default=config.SEED)
    a = p.parse_args(argv)
    cap = None if a.cap <= 0 else a.cap

    splits = {c: split(clean_prices(load_city(c, a.data_dir), cap), a.seed) for c in a.train_cities}
    target = clean_prices(load_city(a.target_city, a.data_dir), cap)

    # training sets: each city alone, then all together
    setups = {c.upper(): [splits[c][0]] for c in a.train_cities}
    setups[" + ".join(c.upper() for c in a.train_cities)] = [splits[c][0] for c in a.train_cities]

    rows_t6 = []
    config.RESULTS_DIR.mkdir(exist_ok=True)
    for label_name, frames in setups.items():
        # Table 6: evaluate on the test split of every training city
        own_tests = {f"test_{c}": splits[c][2] for c in a.train_cities}
        # Table 7: evaluate on the unseen target city
        eval_sets = {**own_tests, "target": target}
        res, preds = _fit_eval(frames, eval_sets, a.features, a.label, a.model, a.seed)
        row = {"trained_on": label_name, "train_r2": res["train"]["r2"], "train_mse": res["train"]["mse"]}
        for c in a.train_cities:
            row[f"{c}_test_r2"], row[f"{c}_test_mse"] = res[f"test_{c}"]["r2"], res[f"test_{c}"]["mse"]
        row[f"{a.target_city}_r2"], row[f"{a.target_city}_mse"] = res["target"]["r2"], res["target"]["mse"]
        rows_t6.append(row)
        yt, yp = preds["target"]
        tag = label_name.lower().replace(" + ", "_and_")
        scatter_plot(yt, yp, config.RESULTS_DIR / f"transfer_{tag}_to_{a.target_city}.png",
                     f"Train: {label_name}  ->  test: {a.target_city.upper()}")
        print(f"train on {label_name:12s} | own-test R2: "
              + ", ".join(f"{c}={res[f'test_{c}']['r2']:.3f}" for c in a.train_cities)
              + f" | {a.target_city} R2={res['target']['r2']:.3f}")

    df = pd.DataFrame(rows_t6)
    df.to_csv(config.RESULTS_DIR / "transfer_results.csv", index=False)
    print("\nSaved", config.RESULTS_DIR / "transfer_results.csv")
    return df


if __name__ == "__main__":
    main()
