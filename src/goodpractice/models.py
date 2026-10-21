"""The two model classes used throughout: a linear baseline and a boosted tree."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = {"linear": "Linear (ridge)", "gbm": "Gradient-boosted trees"}


def make_model(kind: str = "gbm", seed: int = 0, cv_gap: int = 0):
    """Ridge with forward-chaining (time-blocked) alpha selection, or a boosted tree.

    The scaler sits inside the pipeline so it is fitted on training data only
    (avoids leakage type L1.2, preprocessing on all data).

    Ridge alpha is chosen with TimeSeriesSplit (5 folds, `cv_gap` samples
    between each training and validation block), not with RidgeCV's default
    efficient leave-one-out. On autocorrelated data, leave-one-out validates
    each sample against its own neighbours, so it favours too little
    regularisation; with collinear inputs (air valves -> jg) the coefficients
    then become unstable. Hyper-parameter selection can leak too.
    Rows must be in time order.
    """
    if kind == "linear":
        cv = TimeSeriesSplit(n_splits=5, gap=cv_gap)
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15), cv=cv))
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    raise ValueError(f"unknown model kind {kind!r}")


def n_parameters(est) -> int:
    """Rough size: coefficients + intercept for ridge; nodes (leaf values and
    split thresholds) for boosted trees. A proxy for what has to be maintained."""
    if hasattr(est, "_predictors"):
        return int(sum(len(tree.nodes) for trees in est._predictors for tree in trees))
    return int(np.size(est[-1].coef_) + 1)
