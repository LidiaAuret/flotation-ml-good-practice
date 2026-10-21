"""Train/test splits for time-ordered process data.

Kapoor & Narayanan (2023, Patterns 4:100804) list non-independence between
training and test samples (L3.2) and temporal leakage (L3.1) among the most
common leakage types. With 1-min process data, a random split puts the test
sample at 10:01 next to training samples at 10:00 and 10:02, so the test set
measures interpolation, not prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Random split of rows. Shown for contrast; not recommended for time series."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int((1 - test_frac) * len(df))
    return df.iloc[np.sort(idx[:cut])], df.iloc[np.sort(idx[cut:])]


def blocked_split(df: pd.DataFrame, time_col: str, train_end: float,
                  test_start: float, test_end: float | None = None):
    """Contiguous blocks: train on time < train_end, test on [test_start, test_end).

    Leave a gap (test_start > train_end) of at least a few process time
    constants so that the first test samples are not near-copies of the last
    training samples.
    """
    if test_start < train_end:
        raise ValueError("test block must start after the training block ends")
    t = df[time_col]
    train = df[t < train_end]
    test = df[t >= test_start] if test_end is None else df[(t >= test_start) & (t < test_end)]
    return train, test


def lag1_autocorr(x) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))


def n_eff_ratio(rho):
    """Effective sample size ratio for an AR(1) series: n_eff/n = (1-rho)/(1+rho).

    See e.g. Bayley & Hammersley (1946), J. R. Stat. Soc. Suppl. 8(2):184-197.
    At rho = 0.99, 30 000 one-minute samples carry the information of ~150
    independent ones.
    """
    rho = np.asarray(rho, float)
    return (1 - rho) / (1 + rho)
