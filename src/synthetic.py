"""Generate small synthetic Airbnb-like CSVs so the pipeline can be tested/demoed offline.

    python -m src.synthetic                  # -> data/sample/{nyc,paris,berlin}.csv

The data mimics the Kaggle listings schema (money as "$1,234.00" strings, list-valued
amenities, free-text columns, dates) with a price that really depends on the features,
so models learn something. It is NOT real data - never report results from it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_CITY = {  # name: (base log-price, lat, lon, city-specific extra column)
    "nyc": (4.7, 40.73, -73.95, "neighbourhood_group_cleansed"),
    "paris": (4.5, 48.86, 2.35, None),
    "berlin": (4.0, 52.52, 13.40, None),
}
_PROPERTY = ["Apartment", "House", "Loft", "Condominium", "Townhouse", "Bed and breakfast"]
_ROOM = ["Entire home/apt", "Private room", "Shared room"]
_AMENITIES = ["Wifi", "TV", "Kitchen", "Heating", "Washer", "Dryer", "Elevator", "Air conditioning",
              "Hair dryer", "Iron", "Laptop friendly workspace", "Gym", "Pool", "Doorman"]
_LUXURY = ["luxurious", "spacious", "penthouse", "stunning", "designer", "terrace", "view"]
_BUDGET = ["cozy", "small", "simple", "budget", "basic", "quiet", "shared"]
_NEUTRAL = ["apartment", "close", "metro", "station", "bedroom", "bright", "located", "center",
            "street", "kitchen", "living", "room", "welcome", "guests", "park", "cafe"]


def make_city(city: str, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng({"nyc": 1, "paris": 2, "berlin": 3}[city] + seed)
    base, lat0, lon0, extra = _CITY[city]
    room = rng.choice(_ROOM, n, p=[0.55, 0.40, 0.05])
    accommodates = rng.integers(1, 9, n)
    bedrooms = np.clip(np.round(accommodates / 2 + rng.normal(0, 0.6, n)), 0, 6)
    bathrooms = np.clip(np.round(bedrooms / 1.5 + rng.normal(0.5, 0.3, n), 1), 0.5, 4)
    beds = np.clip(np.round(accommodates / 1.8 + rng.normal(0, 0.5, n)), 1, 8)
    lat = lat0 + rng.normal(0, 0.04, n)
    lon = lon0 + rng.normal(0, 0.05, n)
    centrality = -np.hypot(lat - lat0, lon - lon0) * 6
    rating = np.clip(rng.normal(94, 6, n), 60, 100)
    n_amen = rng.integers(3, len(_AMENITIES), n)
    luxury = rng.random(n) < 0.25
    log_price = (base + 0.18 * accommodates + 0.15 * bedrooms + 0.08 * bathrooms
                 + np.where(room == "Entire home/apt", 0.35, np.where(room == "Shared room", -0.5, 0.0))
                 + centrality + 0.02 * n_amen + 0.004 * (rating - 90) + 0.4 * luxury
                 + rng.normal(0, 0.25, n))
    price = np.round(np.exp(log_price))

    def text(i):
        words = list(rng.choice(_NEUTRAL, 12)) + list(rng.choice(_LUXURY if luxury[i] else _BUDGET, 3))
        rng.shuffle(words)
        return " ".join(words)

    amen = [",".join(rng.choice(_AMENITIES, k, replace=False)) for k in n_amen]
    ver = [str(list(rng.choice(["email", "phone", "reviews", "work_email", "government_id"],
                               rng.integers(1, 5), replace=False))) for _ in range(n)]
    hs = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 2500, n), unit="D")
    fr = hs + pd.to_timedelta(rng.integers(30, 600, n), unit="D")
    lr = fr + pd.to_timedelta(rng.integers(0, 700, n), unit="D")

    df = pd.DataFrame({
        "id": np.arange(n) + 1000 * (list(_CITY).index(city) + 1),
        "name": [text(i) for i in range(n)],
        "summary": [text(i) for i in range(n)],
        "description": [text(i) for i in range(n)],
        "transit": [text(i) for i in range(n)],
        "neighborhood_overview": [text(i) for i in range(n)],
        "host_since": hs.strftime("%Y-%m-%d"),
        "first_review": fr.strftime("%Y-%m-%d"),
        "last_review": lr.strftime("%Y-%m-%d"),
        "property_type": rng.choice(_PROPERTY, n),
        "room_type": room,
        "bed_type": rng.choice(["Real Bed", "Futon", "Couch"], n, p=[0.93, 0.04, 0.03]),
        "cancellation_policy": rng.choice(["flexible", "moderate", "strict_14_with_grace_period"], n),
        "host_is_superhost": rng.choice(["t", "f"], n, p=[0.2, 0.8]),
        "instant_bookable": rng.choice(["t", "f"], n),
        "neighbourhood_cleansed": [f"{city}_district_{k}" for k in rng.integers(0, 15, n)],
        "accommodates": accommodates, "bedrooms": bedrooms, "bathrooms": bathrooms, "beds": beds,
        "guests_included": np.minimum(accommodates, rng.integers(1, 4, n)),
        "latitude": lat, "longitude": lon,
        "security_deposit": [f"${int(x):,}.00" if rng.random() < 0.5 else np.nan for x in price * 2],
        "cleaning_fee": [f"${int(x):,}.00" if rng.random() < 0.7 else np.nan for x in price * 0.3],
        "extra_people": [f"${int(x)}.00" for x in rng.integers(0, 30, n)],
        "minimum_nights": rng.integers(1, 10, n), "maximum_nights": rng.integers(30, 1125, n),
        "availability_30": rng.integers(0, 30, n), "availability_365": rng.integers(0, 365, n),
        "number_of_reviews": rng.poisson(20, n), "reviews_per_month": np.round(rng.gamma(2, 0.8, n), 2),
        "review_scores_rating": np.round(rating), "review_scores_value": np.round(np.clip(rating / 10 + rng.normal(0, 0.5, n), 2, 10)),
        "amenities": ["{" + a + "}" for a in amen],
        "host_verifications": ver,
        "price": [f"${int(p):,}.00" for p in price],
    })
    if extra:
        df[extra] = rng.choice(["Manhattan", "Brooklyn", "Queens", "Bronx"], n)
    # a few missing values / junk like the real files
    for c in ["bedrooms", "review_scores_rating", "reviews_per_month"]:
        df.loc[rng.random(n) < 0.08, c] = np.nan
    df.loc[rng.random(n) < 0.05, ["first_review", "last_review"]] = np.nan
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(config.DATA_DIR / "sample"))
    p.add_argument("--n", type=int, default=1500, help="listings per city")
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for city in config.CITIES:
        make_city(city, a.n).to_csv(out / f"{city}.csv", index=False)
        print("wrote", out / f"{city}.csv")


if __name__ == "__main__":
    main()
