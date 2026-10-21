# %% [markdown]
# # 03 Explanations describe the model, not the plant
#
# Compare three answers to "what happens to recovery if I raise input X by one
# standard deviation?":
#   * truth from simulator step tests (data/rougher_bank_step_tests.csv)
#   * standardised ridge coefficients
#   * SHAP (TreeExplainer) slope for the boosted tree, times the input std
#   * the same SHAP slope for a boosted tree trained on causes only
#
# In closed loop, inputs move together (the level controller moves the tail
# valves; air sets jg, which sets bubble size). Mediators such as d32 and jg
# absorb effects that physically belong to the air valves and frother, and
# correlated disturbances get credit for each other. Neither ridge nor SHAP
# can separate these from passive data; a step test can. Dropping the
# mediators (a cause-and-effect graph: physics in the problem framing) moves
# the SHAP slopes towards the truth, at a cost in test R².

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

from common import CAUSES, CV_GAP, DATA, FIGS, LEGIT, NICE, ORE_CHANGE_DAY, TARGET, load
from goodpractice import make_model
from goodpractice.style import CAT, apply, save

apply()
df = load()
train = df[df.day < 10]  # the design-B training block from 02
test = df[(df.time_h >= 10.5 * 24) & (df.time_h < ORE_CHANGE_DAY * 24)]  # design-B test block
truth = pd.read_csv(DATA / "rougher_bank_step_tests.csv").set_index("feature")
stepped = list(truth.index)

ridge = make_model("linear", cv_gap=CV_GAP).fit(train[LEGIT], train[TARGET])
ridge_effect = pd.Series(ridge[-1].coef_, index=LEGIT)  # per 1 std (scaled inputs)

sample = train.sample(3000, random_state=0)


def shap_slopes(features):
    """Fit a boosted tree on `features`; return SHAP slope per std and test R²."""
    model = make_model("gbm").fit(train[features], train[TARGET])
    sv = shap.TreeExplainer(model).shap_values(sample[features])
    slopes = pd.Series(
        [np.polyfit(sample[c], sv[:, j], 1)[0] * train[c].std() for j, c in enumerate(features)], index=features)
    return slopes, r2_score(test[TARGET], model.predict(test[features]))


shap_effect, r2_all = shap_slopes(LEGIT)
shap_causes, r2_causes = shap_slopes(CAUSES)

table = pd.DataFrame({"truth (step test)": truth.true_effect_per_std,
                      "ridge": ridge_effect[stepped], "SHAP (GBM)": shap_effect[stepped],
                      "SHAP (GBM, causes only)": shap_causes[stepped]})
table.index = [NICE[c] for c in table.index]
print(table.round(2).to_string())
print(f"\nGBM test R2 (days 10.5-14): all inputs {r2_all:.2f}, causes only {r2_causes:.2f}")
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
print("corr(air 1-2, jg) =", round(df[["air_valve_cells1to2_pct", "gas_velocity_jg_mean_cms"]].corr().iloc[0, 1], 3))

# %%
fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.arange(len(table))
h = 0.2
labels = list(table.columns)
labels[2] += f" (test R² {r2_all:.2f})"
labels[3] += f" (test R² {r2_causes:.2f})"
for i, col in enumerate(table.columns):
    ax.barh(y + (i - 1.5) * h, table[col], h, color=["#0b0b0b", CAT[0], CAT[1], CAT[2]][i], label=labels[i])
ax.set_yticks(y, table.index)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in recovery (% points) per +1 std of input")
ax.legend(loc="lower left", fontsize=9)
ax.set_title("Only the step test is causal; dropping mediators moves SHAP towards it", loc="left")
save(fig, "03_explanations", FIGS)
table.to_csv(FIGS / "03_effects.csv")
