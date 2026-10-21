# %% [markdown]
# # 02 Same data, same models, four evaluation designs
#
# A  random 80/20 split of 1-min samples (Kapoor & Narayanan L3.2)
# B  blocked time split with a 12 h gap, same ore
# C  train before the ore change, test after it
# D  as B, plus same-time concentrate and tailings assays as inputs (L2)
#
# Two model classes: ridge regression (9 parameters) and gradient-boosted
# trees (tens of thousands of nodes). Is the complex model worth it?
# The second part fits each model once on a random split of days 0-10 and
# follows it, unchanged, through the next days and the ore change.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CV_GAP, FIGS, LEGIT, ORE_CHANGE_DAY, TARGET, designs, load
from goodpractice import make_model, n_parameters, scores
from goodpractice.models import LABELS
from goodpractice.style import CAT, apply, save

apply()
df = load()
rows = []
for key, (label, train, test, feats) in designs(df).items():
    for kind in ["linear", "gbm"]:
        m = make_model(kind, cv_gap=CV_GAP).fit(train[feats], train[TARGET])
        s = scores(test[TARGET], m.predict(test[feats]))
        rows.append({"design": key, "label": label.replace("\n", " "), "model": kind,
                     "n_train": len(train), "n_test": len(test),
                     "n_parameters": n_parameters(m), **s})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
keys = list(designs(df).keys())
labels = [designs(df)[k][0] for k in keys]
w = 0.38
for i, kind in enumerate(["linear", "gbm"]):
    r2 = [res.query("design == @k and model == @kind").r2.item() for k in keys]
    bars = ax.bar(np.arange(len(keys)) + (i - 0.5) * w, r2, w, color=CAT[i], label=LABELS[kind])
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.set_xticks(np.arange(len(keys)), labels)
ax.set_ylabel("Test R² (recovery)")
ax.set_ylim(min(0, res.r2.min() - 0.05), 1.18)
ax.legend(loc="upper center", ncol=2)
ax.set_title("The evaluation design changes the answer more than the model does", loc="left")
save(fig, "02_evaluation_designs", FIGS)
res.to_csv(FIGS / "02_results.csv", index=False)

# %% [markdown]
# ## Fit once, report, deploy
#
# A common workflow: fit on the data at hand with a random 80/20 split, report
# the test score, then use the model on the weeks that follow. Here each model
# is fitted once on a random 80 % of days 0-10 and never retrained. Its
# random-split test (the other 20 %, interleaved in time with the training
# minutes) is the reported score; the same model then meets days 10-14 (same
# ore, straight after the fit) and days 14-21 (after the ore change).

# %%
from goodpractice import random_split

early = df[df.time_h < 10 * 24]
fit_tr, fit_te = random_split(early, 0.2, seed=0)
periods = {
    "Random-split test\n(days 0-10)": fit_te,
    "Later: same ore\n(days 10-14)": df[(df.time_h >= 10 * 24) & (df.time_h < ORE_CHANGE_DAY * 24)],
    "Later: after ore change\n(days 14-21)": df[df.time_h >= ORE_CHANGE_DAY * 24],
}
rows, preds = [], {}
for kind in ["linear", "gbm"]:
    m = make_model(kind, cv_gap=CV_GAP).fit(fit_tr[LEGIT], fit_tr[TARGET])
    for name, part in periods.items():
        p = m.predict(part[LEGIT])
        preds[kind, name] = p
        rows.append({"model": kind, "period": name.replace("\n", " "), "n": len(part),
                     **scores(part[TARGET], p)})
fit_once = pd.DataFrame(rows)
print(fit_once.round(3).to_string(index=False))

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True, sharey=True)
lo, hi = np.floor(df[TARGET].quantile(0.002)), np.ceil(df[TARGET].quantile(0.998))
for ax, (name, part) in zip(axes, periods.items()):
    r = fit_once[(fit_once.model == "gbm") & (fit_once.period == name.replace("\n", " "))].iloc[0]
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--")
    ax.scatter(part[TARGET], preds["gbm", name], s=3, alpha=0.15, color=CAT[1], edgecolors="none", rasterized=True)
    ax.text(0.04, 0.96, f"R² = {r.r2:.2f}\nRMSE = {r.rmse:.2f}\nn = {r.n}", transform=ax.transAxes, va="top")
    ax.set_title(name, loc="left", fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual recovery (%)")
axes[0].set_ylabel("Predicted recovery (%)")
fig.suptitle("One boosted-tree model, fitted once: the reported score does not survive the next days",
             x=0.01, ha="left", fontweight="bold")
save(fig, "02_fit_once", FIGS)
fit_once.to_csv(FIGS / "02_fit_once.csv", index=False)
