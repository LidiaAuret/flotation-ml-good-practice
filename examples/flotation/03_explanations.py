# %% [markdown]
# # 03 Explanations describe the model, not the plant
#
# Compare three answers to "what happens to recovery if I raise input X by one
# standard deviation?":
#   * truth from simulator step tests (data/rougher_bank_step_tests.csv)
#   * standardised ridge coefficients
#   * SHAP (TreeExplainer) slope for the boosted tree, times the input std
#
# In closed loop, inputs move together (the level controller moves the tail
# valves; air sets jg, which sets bubble size). Mediators such as d32 and jg
# absorb effects that physically belong to the air valves and frother, and
# correlated disturbances get credit for each other. Neither ridge nor SHAP
# can separate these from passive data; a step test can.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import CV_GAP, DATA, FIGS, LEGIT, NICE, ORE_CHANGE_DAY, TARGET, load
from goodpractice import make_model
from goodpractice.style import CAT, apply, save

apply()
df = load()
train = df[df.day < 10]  # the design-B training block from 02
truth = pd.read_csv(DATA / "rougher_bank_step_tests.csv").set_index("feature")
stepped = list(truth.index)

ridge = make_model("linear", cv_gap=CV_GAP).fit(train[LEGIT], train[TARGET])
ridge_effect = pd.Series(ridge[-1].coef_, index=LEGIT)  # per 1 std (scaled inputs)

gbm = make_model("gbm").fit(train[LEGIT], train[TARGET])
sample = train.sample(3000, random_state=0)
sv = shap.TreeExplainer(gbm).shap_values(sample[LEGIT])
shap_effect = pd.Series(
    [np.polyfit(sample[c], sv[:, j], 1)[0] * train[c].std() for j, c in enumerate(LEGIT)], index=LEGIT)

table = pd.DataFrame({"truth (step test)": truth.true_effect_per_std,
                      "ridge": ridge_effect[stepped], "SHAP (GBM)": shap_effect[stepped]})
table.index = [NICE[c] for c in table.index]
print(table.round(2))
print("\nInputs without a step test (mediators / controlled):")
print(pd.DataFrame({"ridge": ridge_effect, "SHAP": shap_effect}).drop(stepped).round(2))

# %% Stability: refit the ridge on days 0-14 instead of 0-10.
# Air valve openings and jg carry overlapping information (jg follows the
# air valves), so the split of credit between them is poorly determined.
# With leave-one-out alpha selection (sklearn's RidgeCV default) the air and
# jg coefficients swing to large opposite-signed values and change with the
# rounding of the CSV; time-blocked CV (used here) is far more stable.
# Check the conditioning of your inputs (correlations, VIF) before reading
# coefficients as effects.
ridge14 = make_model("linear", cv_gap=CV_GAP).fit(df[df.day < ORE_CHANGE_DAY][LEGIT], df[df.day < ORE_CHANGE_DAY][TARGET])
print(pd.DataFrame({"ridge, days 0-10": ridge_effect,
                    "ridge, days 0-14": pd.Series(ridge14[-1].coef_, index=LEGIT)}).round(2))
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
loo = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(train[LEGIT], train[TARGET])
print("\nleave-one-out alpha:", loo[-1].alpha_); print(pd.Series(loo[-1].coef_, index=LEGIT).round(1))
print("corr(air 1-4, jg) =", round(df[["air_valve_cells1to4_pct", "gas_velocity_jg_mean_cms"]].corr().iloc[0, 1], 3))

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(table))
h = 0.26
for i, col in enumerate(table.columns):
    ax.barh(y + (1 - i) * h, table[col], h, color=["#0b0b0b", CAT[0], CAT[1]][i], label=col)
ax.set_yticks(y, table.index)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in recovery (% points) per +1 std of input")
ax.legend(loc="lower right")
ax.set_title("Same data, three 'effects': only the step test is causal", loc="left")
save(fig, "03_explanations", FIGS)
table.to_csv(FIGS / "03_effects.csv")
