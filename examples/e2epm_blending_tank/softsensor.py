# %% [markdown]
# # Soft sensor for tank concentration: physics, data, or both?
#
# Data: ../../data/e2epm_blending_tank.csv from simulate.py (E2EPM blending
# tank, Auret & Louw 2023; 100 s samples, 28 days). The concentration
# analyser C_meas is the label. It drifts when its fault occurs, so every
# model is also scored against the simulator's truth_C, which no plant has.
#
# Questions:
#  1. Does a random split flatter the models here too?
#  2. Does a zero-parameter mass balance beat ridge and boosted trees?
#  3. Does adding the mass balance as a feature (hybrid) help?
#  4. What happens if the controller output is used as an input?

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from goodpractice import blocked_split, lag1_autocorr, make_model, random_split, scores  # noqa: E402
from goodpractice.style import CAT, GREY, apply, save  # noqa: E402
from tank_physics import mass_balance_estimate  # noqa: E402

apply()
FIGS = ROOT / "figures" / "e2epm_blending_tank"
df = pd.read_csv(ROOT / "data" / "e2epm_blending_tank.csv")
running = (df.regime == "Running").to_numpy()

# %% Physics estimate, from measured inputs and (for reference) from true inputs
df["C_physics"] = mass_balance_estimate(df.F0_meas, df.C0_meas, df.FW_meas, df.L_meas, running, df.C_meas)
df["C_physics_true_inputs"] = mass_balance_estimate(df.truth_F0, df.truth_C0, df.truth_FW, df.truth_L,
                                                    running, df.truth_C)

run = df[running & (df.time_h > 2)].copy()
print(f"running samples: {len(run)};  lag-1 autocorrelation of C_meas: {lag1_autocorr(run.C_meas):.2f}")
for c in ["F0", "FW", "F", "C0", "L", "C"]:
    noise = (run[f"{c}_meas"] - run[f"truth_{c}"]).std()
    print(f"  {c:>2}: sensor noise std / process std = {noise / run[f'truth_{c}'].std():.2f}")

# %% Splits
# Random: 80/20 of running samples.
# Blocked: train before the first fault (valve sticks at ~309 h), test after the
# first repair (~345 h). The test block contains a second C-sensor drift,
# so labels in the test block are partly wrong.
A_tr, A_te = random_split(run, 0.2, seed=0)
B_tr, B_te = blocked_split(run, "time_h", train_end=300, test_start=345)
FAULT = ["fault_C_sensor_drift", "fault_water_valve_stuck"]
B_te_ok = B_te[(B_te[FAULT] == 0).all(axis=1)]  # blocked test block without fault periods

PROCESS = ["C0_meas", "F0_meas", "FW_meas", "F_meas", "L_meas"]
candidates = {
    "Training mean (no model)": "mean",
    "Mass balance (0 parameters)": None,
    "Ridge": ("linear", PROCESS),
    "Boosted trees": ("gbm", PROCESS),
    "Ridge + mass balance": ("linear", PROCESS + ["C_physics"]),
    "Boosted trees + mass balance": ("gbm", PROCESS + ["C_physics"]),
    "Ridge + controller output": ("linear", PROCESS + ["water_valve_command"]),
}

rows, preds = [], {}
for name, spec in candidates.items():
    fitted = {}
    for split, (tr, te) in {"random": (A_tr, A_te), "blocked": (B_tr, B_te),
                            "blocked, no faults": (B_tr, B_te_ok)}.items():
        if spec == "mean":
            p = np.full(len(te), tr.C_meas.mean())
        elif spec is None:
            p = te.C_physics.to_numpy()
        else:
            kind, feats = spec
            key = id(tr)
            if key not in fitted:
                fitted[key] = make_model(kind, cv_gap=36).fit(tr[feats], tr.C_meas)
            p = fitted[key].predict(te[feats])
        preds[(name, split)] = p
        rows.append({"model": name, "split": split,
                     "r2_vs_analyser": scores(te.C_meas, p)["r2"],
                     "r2_vs_truth": scores(te.truth_C, p)["r2"],
                     "rmse_vs_truth": scores(te.truth_C, p)["rmse"]})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
print(f"\nanalyser itself vs truth, blocked test block: R2 = {scores(B_te.truth_C, B_te.C_meas)['r2']:.3f}")
print(f"mass balance with TRUE inputs (structure check): R2 = "
      f"{scores(run.truth_C, run.C_physics_true_inputs)['r2']:.3f}")
print(f"water_valve_command range: train {B_tr.water_valve_command.min():.1f}..{B_tr.water_valve_command.max():.1f}, "
      f"test {B_te.water_valve_command.min():.1f}..{B_te.water_valve_command.max():.1f}")
FIGS.mkdir(parents=True, exist_ok=True)
res.to_csv(FIGS / "softsensor_results.csv", index=False)

# %% Figure 1: RMSE against truth for the three test sets
# R2 is not comparable across test sets with different variance: removing
# the fault periods lowers R2 (less variance to explain) while the error
# shrinks. RMSE in physical units, next to a no-model baseline, is clearer.
show = [m for m in candidates if m != "Ridge + controller output"]
fig, ax = plt.subplots(figsize=(10, 4.5))
w = 0.27
for i, split in enumerate(["random", "blocked, no faults", "blocked"]):
    v = [res.query("model == @m and split == @split").rmse_vs_truth.item() for m in show]
    bars = ax.bar(np.arange(len(show)) + (i - 1) * w, v, w, color=CAT[i], label=split)
    ax.bar_label(bars, fmt="%.3f", fontsize=7)
ax.set_xticks(np.arange(len(show)), [m.replace(" + ", "\n+ ").replace(" (", "\n(") for m in show], fontsize=9)
ax.set_ylabel("RMSE against true C (kg/m³)")
ax.legend(loc="upper right", title="test set")
ax.set_title("Blending-tank soft sensor: split design and physics features", loc="left")
save(fig, "softsensor_rmse", FIGS)

# %% Figure 2: a window of the blocked test block, including sensor drift
win = B_te[(B_te.time_h > 640) & (B_te.time_h < 700)]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(win.time_h, win.C_meas, color=GREY, lw=0.7, label="analyser (label)")
ax.plot(win.time_h, win.truth_C, color="black", lw=1.2, label="true concentration")
for j, name in enumerate(["Ridge", "Ridge + mass balance"]):
    p = pd.Series(preds[(name, "blocked")], index=B_te.index).loc[win.index]
    ax.plot(win.time_h, p, color=CAT[j + 2], lw=1, label=name)
drift = win[win.fault_C_sensor_drift > 0]
if len(drift):
    ax.axvspan(drift.time_h.min(), drift.time_h.max(), color=CAT[3], alpha=0.12, lw=0, label="analyser drifting")
ax.set_xlabel("time (h)")
ax.set_ylabel("C (kg/m³)")
stuck = win[win.fault_water_valve_stuck > 0]
if len(stuck):
    ax.axvline(stuck.time_h.min(), color=CAT[7], lw=1, ls="--", label="water valve sticks open")
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax.set_title("A fault moves the process outside the training range; the drifting analyser then mislabels it",
             loc="left", fontsize=11)
save(fig, "softsensor_timeseries", FIGS)
