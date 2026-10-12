"""Smoke tests on synthetic data (no Kaggle files needed).  Run with:  pytest -q"""
import numpy as np
import pytest

from src import config
from src.data_loader import clean_prices, inverse_label, load_city, parse_money, split, transform_label
from src.features import FeatureBuilder, parse_feature_set
from src.models import get_model
from src.synthetic import make_city
import pandas as pd


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for c in config.CITIES:
        make_city(c, 1500).to_csv(d / f"{c}.csv", index=False)
    return d


def test_parse_money():
    s = parse_money(pd.Series(["$1,234.00", "$5.50", None, "abc"]))
    assert s.iloc[0] == 1234.0 and s.iloc[1] == 5.5 and s.iloc[2:].isna().all()


def test_label_roundtrip():
    p = np.array([50.0, 120.0, 400.0])
    for how in ("log", "sqrt", "none"):
        assert np.allclose(inverse_label(transform_label(p, how), how), p)


def test_split_sizes(frames):
    df = clean_prices(load_city("nyc", frames))
    tr, va, te = split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert abs(len(tr) / len(df) - 0.7) < 0.02 and abs(len(te) / len(df) - 0.1) < 0.02
    assert df["price"].max() <= config.PRICE_CAP


def test_parse_feature_set():
    assert parse_feature_set("C+O+T") == {"C", "O", "T"}
    with pytest.raises(ValueError):
        parse_feature_set("X")


@pytest.mark.parametrize("groups", ["C", "C+O", "C+O+T", "D"])
def test_feature_shapes_consistent(frames, groups):
    df = clean_prices(load_city("nyc", frames))
    tr, _, te = split(df)
    fb = FeatureBuilder(groups).fit(tr)
    a, b = fb.transform(tr), fb.transform(te)
    assert a.shape[1] == b.shape[1] == len(fb.feature_names_)
    assert not np.isnan(a).any() and not np.isnan(b).any()


def test_interactions_add_features(frames):
    tr, _, _ = split(clean_prices(load_city("nyc", frames)))
    plain = FeatureBuilder("C").fit(tr)
    inter = FeatureBuilder("C", interactions=True).fit(tr)
    assert inter.transform(tr).shape[1] > plain.transform(tr).shape[1]


@pytest.mark.parametrize("model", ["linear", "knn", "rf", "xgb", "nn"])
def test_models_learn_something(frames, model):
    tr, _, te = split(clean_prices(load_city("nyc", frames)))
    fb = FeatureBuilder("C+O").fit(tr)
    m = get_model(model).fit(fb.transform(tr), transform_label(tr["price"], "log"))
    from sklearn.metrics import r2_score
    r2 = r2_score(transform_label(te["price"], "log"), m.predict(fb.transform(te)))
    assert r2 > 0.2, f"{model} R2={r2:.2f}"


def test_unseen_city_has_missing_columns(frames):
    """A builder fit on NYC+Paris must still transform Berlin (different columns)."""
    tr = pd.concat([split(clean_prices(load_city(c, frames)))[0] for c in ("nyc", "paris")], ignore_index=True)
    berlin = clean_prices(load_city("berlin", frames))
    fb = FeatureBuilder("C+O").fit(tr)
    assert fb.transform(berlin).shape[1] == fb.transform(tr).shape[1]


def test_cli_end_to_end(frames, monkeypatch, tmp_path):
    from src import train, transfer
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    res = train.main(["--city", "nyc", "--models", "linear", "xgb", "--features", "C+O+T",
                      "--data-dir", str(frames), "--no-plots"])
    assert set(res["model"]) == {"linear", "xgb"}
    assert (tmp_path / "models" / "nyc_xgb.joblib").exists()
    t = transfer.main(["--data-dir", str(frames)])
    assert len(t) == 3
