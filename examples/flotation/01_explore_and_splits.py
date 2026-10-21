# %% [markdown]
# # 01 Explore the data and see why random splits mislead
#
# Synthetic 8-cell rougher bank, 1-min samples, 21 days, ore change at day 14.
# See data/DATA_CARD.md for every column. `truth_*` and `hidden_*` columns are
# for evaluation and discussion only: never use them as model inputs.

# %%
import matplotlib.pyplot as plt
import numpy as np

from common import FIGS, ORE_CHANGE_DAY, TARGET, load
from goodpractice import lag1_autocorr, n_eff_ratio, random_split
from goodpractice.style import CAT, GREY, apply, save

apply()
df = load()
print(df.shape)
print(df.describe().T[["mean", "std", "min", "max"]].round(3))

# %% Time series of disturbances, manipulated variables and the target
panels = [
    ("Disturbances", ["feed_rate_tph"], "t/h"),
    ("", ["feed_grade_online_pct"], "% Cu"),
    ("Manipulated variables", ["air_valve_cells1to4_pct", "air_valve_cells5to8_pct", "frother_pump_speed_pct"], "%"),
    ("Target", [TARGET, "truth_recovery_pct"], "%"),
]
fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9), sharex=True)
for ax, (title, cols, unit) in zip(axes, panels):
    for i, c in enumerate(cols):
        ax.plot(df.time_h / 24, df[c], lw=0.6, color=CAT[i], label=c)
    ax.axvline(ORE_CHANGE_DAY, color=GREY, ls="--", lw=1)
    ax.set_ylabel(unit)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
axes[-1].set_xlabel("Day (dashed line: ore change)")
save(fig, "01_timeseries", FIGS)

# %% How much independent information is in 30 000 samples?
# Sub-sample the 1-min series and look at lag-1 autocorrelation of recovery.
pre = df[df.day < ORE_CHANGE_DAY]
for step in [1, 5, 15, 60, 240]:
    rho = lag1_autocorr(pre[TARGET].to_numpy()[::step])
    n = len(pre) // step
    print(f"every {step:>3} min: n = {n:>6}, lag-1 rho = {rho:.3f}, "
          f"n_eff ~ {n * n_eff_ratio(rho):,.0f}")

# %% Random split: every test sample has training neighbours one minute away
tr, te = random_split(pre, 0.2, 0)
nearest = np.searchsorted(tr.time_h.to_numpy(), te.time_h.to_numpy())
nearest = np.clip(nearest, 1, len(tr) - 1)
d = np.minimum(np.abs(tr.time_h.to_numpy()[nearest] - te.time_h.to_numpy()),
               np.abs(tr.time_h.to_numpy()[nearest - 1] - te.time_h.to_numpy())) * 60
print(f"random split: median distance from a test sample to its nearest training sample = {np.median(d):.1f} min")
