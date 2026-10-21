from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def scores(y_true, y_pred) -> dict:
    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}
