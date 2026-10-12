"""Feature engineering: continuous (C), categorical/other (O), text (T), date (D).

`FeatureBuilder` is fit on training data only and then applied to validation / test /
unseen-city data, so no information leaks from the evaluation sets. Columns that a city's
CSV does not have are treated as missing, which is what lets us fit one builder on
NYC + Paris and still apply it to Berlin.
"""
from __future__ import annotations

import ast
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from . import config
from .data_loader import parse_money


def _to_tokens(value) -> list[str]:
    """Parse '{TV,"Wifi"}' / "['email', 'phone']" / 'a, b' into a list of tokens."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(t).strip().lower() for t in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            pass
    s = s.strip("{}[]")
    return [t.strip().strip("'\"").lower() for t in re.split(r",", s) if t.strip().strip("'\"")]


def parse_feature_set(spec: str) -> set[str]:
    """'C+O+T' -> {'C','O','T'} (letters C, O, T, D)."""
    parts = {p.strip().upper() for p in spec.replace(",", "+").split("+") if p.strip()}
    bad = parts - {"C", "O", "T", "D"}
    if bad:
        raise ValueError(f"unknown feature group(s) {bad}; use C, O, T, D")
    return parts


class FeatureBuilder:
    def __init__(self, groups="C+O", interactions: bool = False, seed: int = config.SEED):
        self.groups = parse_feature_set(groups) if isinstance(groups, str) else set(groups)
        self.interactions = interactions
        self.seed = seed
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _col(df: pd.DataFrame, name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    def _numeric(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in cols:
            s = self._col(df, c)
            out[c] = parse_money(s) if c in config.MONEY_COLS else pd.to_numeric(s, errors="coerce")
        return out

    # --------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.blocks_: list[str] = []
        if "C" in self.groups:
            self._fit_continuous(df)
        if "O" in self.groups:
            self._fit_categorical(df)
        if "T" in self.groups:
            self._fit_text(df)
        if "D" in self.groups:
            self._fit_dates(df)
        if not self.blocks_:
            raise ValueError("no feature block could be built from this data")
        return self

    def _fit_continuous(self, df):
        present = [c for c in config.CONTINUOUS_COLS if c in df.columns and df[c].notna().any()]
        raw = self._numeric(df, present)
        # drop one of each highly-correlated pair
        corr = raw.corr().abs()
        keep, dropped = [], set()
        for i, c in enumerate(present):
            if c in dropped:
                continue
            keep.append(c)
            for other in present[i + 1:]:
                if other not in dropped and corr.loc[c, other] > config.CORR_THRESHOLD:
                    dropped.add(other)
        self.cont_cols_ = keep
        self.cont_fill_ = {
            c: (0.0 if c in config.ZERO_FILL_COLS else float(raw[c].mean()) if raw[c].notna().any() else 0.0)
            for c in keep
        }
        filled = raw[keep].fillna(self.cont_fill_)
        self.scaler_ = StandardScaler().fit(filled)
        names = list(keep)
        if self.interactions:
            self.poly_ = PolynomialFeatures(2, interaction_only=True, include_bias=False)
            self.poly_.fit(self.scaler_.transform(filled))
            names = list(self.poly_.get_feature_names_out(keep))
        self.feature_names_ += [f"C:{n}" for n in names]
        self.blocks_.append("C")

    def _fit_categorical(self, df):
        self.cat_cols_ = [c for c in config.CATEGORICAL_COLS if c in df.columns and df[c].notna().any()]
        if self.cat_cols_:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
            self.ohe_.fit(df[self.cat_cols_].astype(str))
            self.feature_names_ += [f"O:{n}" for n in self.ohe_.get_feature_names_out(self.cat_cols_)]
        self.vocab_: dict[str, list[str]] = {}
        for col, top_n in config.LIST_COLS.items():
            if col in df.columns:
                counts = Counter(t for v in df[col] for t in _to_tokens(v))
                self.vocab_[col] = [t for t, _ in counts.most_common(top_n)]
                self.feature_names_ += [f"O:{col}={t}" for t in self.vocab_[col]]
        self.blocks_.append("O")

    def _fit_text(self, df):
        self.text_models_ = {}
        for col in config.TEXT_COLS:
            if col not in df.columns:
                continue
            docs = df[col].fillna("").astype(str)
            if (docs.str.len() > 0).sum() < 20:
                continue
            tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=config.TFIDF_MAX_FEATURES,
                                    stop_words="english", min_df=2, sublinear_tf=True)
            mat = tfidf.fit_transform(docs)
            k = min(config.SVD_COMPONENTS, mat.shape[1] - 1, mat.shape[0] - 1)
            if k < 1:
                continue
            svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(mat)
            self.text_models_[col] = (tfidf, svd)
            self.feature_names_ += [f"T:{col}_{i}" for i in range(k)]
        self.blocks_.append("T")

    def _fit_dates(self, df):
        self.date_cols_ = [c for c in config.DATE_COLS if c in df.columns]
        parsed = {c: pd.to_datetime(df[c], errors="coerce") for c in self.date_cols_}
        self.date_origin_ = min((p.min() for p in parsed.values() if p.notna().any()), default=pd.Timestamp("2008-01-01"))
        self.date_mean_ = {}
        self.date_std_ = {}
        for c, p in parsed.items():
            days = (p - self.date_origin_).dt.days
            self.date_mean_[c] = float(days.mean()) if days.notna().any() else 0.0
            std = float(days.std()) if days.notna().sum() > 1 else 1.0
            self.date_std_[c] = std if std and not np.isnan(std) else 1.0
        self.feature_names_ += [f"D:{c}" for c in self.date_cols_]
        self.blocks_.append("D")

    # --------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if "C" in self.blocks_:
            raw = self._numeric(df, self.cont_cols_).fillna(self.cont_fill_)
            x = self.scaler_.transform(raw[self.cont_cols_])
            parts.append(self.poly_.transform(x) if self.interactions else x)
        if "O" in self.blocks_:
            if getattr(self, "cat_cols_", None):
                data = df.reindex(columns=self.cat_cols_)
                parts.append(self.ohe_.transform(data.astype(str)))
            for col, vocab in self.vocab_.items():
                idx = {t: i for i, t in enumerate(vocab)}
                mat = np.zeros((len(df), len(vocab)))
                if col in df.columns:
                    for r, v in enumerate(df[col]):
                        for t in _to_tokens(v):
                            if t in idx:
                                mat[r, idx[t]] = 1.0
                parts.append(mat)
        if "T" in self.blocks_:
            for col, (tfidf, svd) in self.text_models_.items():
                docs = self._col(df, col).fillna("").astype(str)
                parts.append(svd.transform(tfidf.transform(docs)))
        if "D" in self.blocks_:
            cols = []
            for c in self.date_cols_:
                days = (pd.to_datetime(self._col(df, c), errors="coerce") - self.date_origin_).dt.days
                days = days.fillna(self.date_mean_[c]).to_numpy(dtype=float)
                cols.append((days - self.date_mean_[c]) / self.date_std_[c])
            if cols:
                parts.append(np.column_stack(cols))
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)
