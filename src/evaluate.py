"""Metrics and plots."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def scatter_plot(y_true, y_pred, path, title="Predicted vs ground truth"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Ground-truth label")
    ax.set_ylabel("Predicted label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
