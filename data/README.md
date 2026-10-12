# Data

The raw Kaggle files are large, so they are **not** committed to Git (see `.gitignore`).
Download them and place them in this folder with exactly these names:

| City   | Kaggle dataset            | Expected file       |
|--------|---------------------------|---------------------|
| NYC    | "Airbnb Open Data in NYC" | `data/nyc.csv`      |
| Paris  | "Airbnb Paris"            | `data/paris.csv`    |
| Berlin | "Berlin Airbnb Data"      | `data/berlin.csv`   |

Use the detailed **listings** table (about 96 columns, including `price`, `accommodates`,
`bedrooms`, `amenities`, `summary`, ...). If a download contains several CSVs, pick the one
with the 96-column listings table and rename it.

You can also point the scripts at any path with `--path`, for example
`python -m src.train --city nyc --path "C:/somewhere/listings.csv"`.

## No Kaggle data yet?

Generate a small synthetic dataset with the same schema so you can test and demo the code:

```bash
python -m src.synthetic          # writes data/sample/{nyc,paris,berlin}.csv
python -m src.train --city nyc --data-dir data/sample
```

Synthetic data is only for checking that the code runs. Report results from the real data.
