"""Loading, cleaning and splitting the Airbnb listing tables."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def parse_money(series: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0 ; numeric input passes through; junk -> NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned.replace("", np.nan), errors="coerce")


def resolve_path(city: str, data_dir: str | Path | None = None, path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(data_dir or config.DATA_DIR) / f"{city}.csv"


def load_city(city: str, data_dir=None, path=None) -> pd.DataFrame:
    """Read one city's listings CSV and attach a numeric `price` column."""
    csv = resolve_path(city, data_dir, path)
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Download the Kaggle file and save it there "
            f"(see data/README.md), or run `python -m src.synthetic` for sample data."
        )
    df = pd.read_csv(csv, low_memory=False)
    if "price" not in df.columns:
        raise ValueError(f"{csv} has no 'price' column - is this the detailed listings table?")
    df["price"] = parse_money(df["price"])
    for col in config.MONEY_COLS:
        if col in df.columns:
            df[col] = parse_money(df[col])
    return df


def clean_prices(df: pd.DataFrame, cap: float | None = config.PRICE_CAP) -> pd.DataFrame:
    """Drop listings with missing / non-positive price and (optionally) price above `cap`."""
    out = df[df["price"].notna() & (df["price"] > 0)]
    if cap is not None:
        out = out[out["price"] <= cap]
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = config.SEED):
    """70/20/10 train/validation/test split."""
    train_frac, val_frac, test_frac = config.SPLIT
    train, rest = train_test_split(df, train_size=train_frac, random_state=seed)
    val, test = train_test_split(rest, train_size=val_frac / (val_frac + test_frac), random_state=seed)
    return tuple(x.reset_index(drop=True) for x in (train, val, test))


# --- label transformation ---------------------------------------------------
def transform_label(price: np.ndarray, how: str) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    if how == "log":
        return np.log(price)
    if how == "sqrt":
        return np.sqrt(price)
    if how in ("none", "raw"):
        return price
    raise ValueError(f"unknown label transform: {how}")


def inverse_label(y: np.ndarray, how: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if how == "log":
        return np.exp(y)
    if how == "sqrt":
        return np.clip(y, 0, None) ** 2
    return y


def prepare_city(city: str, data_dir=None, path=None, cap=config.PRICE_CAP, seed=config.SEED):
    """Convenience: load -> clean -> split. Returns (train, val, test) DataFrames."""
    df = clean_prices(load_city(city, data_dir, path), cap)
    return split(df, seed)
