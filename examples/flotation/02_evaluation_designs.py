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

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CV_GAP, FIGS, TARGET, designs, load
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
