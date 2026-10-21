"""Shared definitions for the flotation examples."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "flotation"

ORE_CHANGE_DAY = 14.0
TARGET = "recovery_assay_pct"
CV_GAP = 720  # samples (12 h) between ridge CV training and validation blocks

# Inputs available on line at prediction time.
LEGIT = [
    "feed_rate_tph", "feed_grade_online_pct",
    "air_valve_cells1to2_pct", "air_valve_cells3to4_pct", "frother_pump_speed_pct",
    "tail_valve_mean_pct", "bubble_d32_mean_mm", "gas_velocity_jg_mean_cms",
]
# Causes only: measured disturbances and operator moves. Tail valves, d32 and jg
# are consequences of these (level control, air -> jg -> bubble size), so a
# what-if model should not hold them fixed.
CAUSES = [
    "feed_rate_tph", "feed_grade_online_pct",
    "air_valve_cells1to2_pct", "air_valve_cells3to4_pct", "frother_pump_speed_pct",
]
# Same-time concentrate and tailings assays: the target is computed from them
# (two-product formula), so using them as inputs is leakage type L2.
LEAKY = LEGIT + ["conc_grade_assay_pct", "tail_grade_assay_pct"]

NICE = {
    "feed_rate_tph": "Feed rate", "feed_grade_online_pct": "Feed grade (on-line)",
    "air_valve_cells1to2_pct": "Air valves, cells 1-2", "air_valve_cells3to4_pct": "Air valves, cells 3-4",
    "frother_pump_speed_pct": "Frother pump", "tail_valve_mean_pct": "Tail valves (mean)",
    "bubble_d32_mean_mm": "Bubble d32 (mean)", "gas_velocity_jg_mean_cms": "Gas velocity jg (mean)",
    "conc_grade_assay_pct": "Conc. grade assay", "tail_grade_assay_pct": "Tail grade assay",
}


def load() -> pd.DataFrame:
    return pd.read_csv(DATA / "rougher_bank_synthetic.csv")


def designs(df: pd.DataFrame, seed: int = 0, train_days: float = 10.0, gap_days: float = 0.5):
    """The four evaluation designs: (label, train, test, features)."""
    from goodpractice import blocked_split, random_split

    pre = df[df.day < ORE_CHANGE_DAY]
    post = df[df.day >= ORE_CHANGE_DAY]
    a_tr, a_te = random_split(pre, 0.2, seed)
    b_tr, b_te = blocked_split(pre, "time_h", train_days * 24, (train_days + gap_days) * 24)
    return {
        "A": ("Random split\n(1-min samples)", a_tr, a_te, LEGIT),
        "B": ("Blocked time split\n(same ore)", b_tr, b_te, LEGIT),
        "C": ("Out-of-time\n(after ore change)", pre, post, LEGIT),
        "D": ("Blocked split +\nsame-time assays (leaky)", b_tr, b_te, LEAKY),
    }
