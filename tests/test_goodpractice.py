import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "e2epm_blending_tank"))

from goodpractice import blocked_split, lag1_autocorr, make_model, n_eff_ratio, random_split  # noqa: E402
from tank_physics import mass_balance_estimate  # noqa: E402


@pytest.fixture
def ts():
    return pd.DataFrame({"t": np.arange(1000.0), "x": np.random.default_rng(0).normal(size=1000)})


def test_random_split_partitions(ts):
    tr, te = random_split(ts, 0.2, 0)
    assert len(tr) + len(te) == len(ts)
    assert set(tr.index).isdisjoint(te.index)


def test_blocked_split_order_and_gap(ts):
    tr, te = blocked_split(ts, "t", 600, 650)
    assert tr.t.max() < 600 <= 650 <= te.t.min()
    with pytest.raises(ValueError):
        blocked_split(ts, "t", 600, 500)


def test_n_eff_ratio():
    assert n_eff_ratio(0.0) == pytest.approx(1.0)
    assert n_eff_ratio(0.99) == pytest.approx(0.01 / 1.99)


def test_lag1_autocorr_ar1():
    rng = np.random.default_rng(1)
    x = np.zeros(20000)
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    assert lag1_autocorr(x) == pytest.approx(0.9, abs=0.02)


def test_make_model_kinds():
    X = np.random.default_rng(0).normal(size=(200, 3))
    y = X @ [1.0, -2.0, 0.5]
    assert make_model("linear").fit(X, y).score(X, y) > 0.99
    with pytest.raises(ValueError):
        make_model("svm")


def test_mass_balance_reaches_steady_state():
    n = 200
    F0, C0, FW, L = (np.full(n, v) for v in (0.005, 1.0, 0.015, 1.0))
    est = mass_balance_estimate(F0, C0, FW, L, np.ones(n, bool), np.zeros(n))
    assert est[-1] == pytest.approx(0.005 / 0.020, rel=1e-6)   # F0 C0 / (F0 + FW)
    assert np.all(np.diff(est) >= 0)                            # monotone approach from 0


def test_data_files_present():
    df = pd.read_csv(ROOT / "data" / "rougher_bank_synthetic.csv", nrows=5)
    assert "recovery_assay_pct" in df and "truth_recovery_pct" in df
