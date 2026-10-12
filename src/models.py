"""Model zoo: baselines (linear, KNN), random forest, XGBoost and the neural network."""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import config

MODEL_NAMES = ("linear", "knn", "rf", "xgb", "nn", "nn_adam")


def _scaled_target(mlp):
    """Standardise the label for the network (the output layer starts near 0, so an
    un-centred log-price of ~4.5 makes SGD converge very slowly). Predictions are
    transformed back automatically, so callers still work on the original label scale."""
    return TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())


def get_model(name: str, seed: int = config.SEED):
    if name == "linear":       # L2-regularised linear regression
        return Ridge(alpha=1.0, random_state=seed)
    if name == "knn":          # unweighted KNN, Euclidean distance
        return KNeighborsRegressor(n_neighbors=5, weights="uniform", n_jobs=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:   # pragma: no cover
            raise ImportError("xgboost is not installed: pip install xgboost") from e
        return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                            random_state=seed, verbosity=0)
    if name == "nn":
        # "Transfer-learning" network from the paper (section 6.2): 2 small hidden layers
        # (64, 32), SGD + Nesterov momentum, L2 = 0.005, early stopping on 10% held-out
        # training data (stop if validation R^2 improves < 1e-4 for 10 iterations).
        return _scaled_target(MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="sgd", learning_rate_init=0.005, momentum=0.9,
            nesterovs_momentum=True, alpha=0.005, batch_size=64,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, tol=1e-4, max_iter=200,
            random_state=seed))
    if name == "nn_adam":
        # First "pioneer" network: 3 hidden layers (128, 512, 64), Adam lr=1e-4, wd=1e-5, batch 64
        return _scaled_target(MLPRegressor(
            hidden_layer_sizes=(128, 512, 64), activation="relu",
            solver="adam", learning_rate_init=1e-4, alpha=1e-5,
            batch_size=64, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, tol=1e-4, max_iter=100,
            random_state=seed))
    raise ValueError(f"unknown model '{name}'. choose from {MODEL_NAMES}")
