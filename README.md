# flotation-ml-good-practice

Open data and small, runnable examples on **evaluating machine-learning models
for process data honestly**: time-aware splits, leakage, baselines, physics
features, and what model explanations can and cannot tell you.

Companion material to the IMPC 2026 keynote by Lidia Auret,
*From promising to proven: Good practice for machine learning in flotation*
([slides](slides/Auret_IMPC2026_keynote_slides.pdf)). The examples are
deliberately simple (ridge regression and gradient-boosted trees in
scikit-learn) so that the evaluation design, not the model, is the focus.

> Everything here is synthetic. No plant data are included.

## What is in the repository

| Path | Content |
|---|---|
| `data/rougher_bank_synthetic.csv` | 21 days of 1-min data from a simulated 4-cell copper rougher bank with concentrate and tailings sumps, with an ore change at day 14. Measured values, plant-style assays **and** the simulator's true values. See [`data/DATA_CARD.md`](data/DATA_CARD.md). |
| `data/rougher_bank_step_tests.csv` | True effect of each input on recovery, from step tests on the simulator: the "ground truth" against which model explanations are checked. |
| `data/e2epm_blending_tank.csv` | 28 days of 100 s data from the end-to-end process monitoring (E2EPM) blending-tank simulator, with sensor drift and valve faults, measured and true values. Regenerate with `examples/e2epm_blending_tank/simulate.py`. |
| `src/goodpractice/` | Thin helpers: random vs blocked splits, effective sample size, the two model classes, scores, plot style. |
| `examples/flotation/` | Three scripts (01-03) on the rougher-bank data. |
| `examples/e2epm_blending_tank/` | Fetch the open-source E2EPM simulator, generate data, build a concentration soft sensor. |
| `tests/` | `pytest` checks for the helpers and the mass-balance soft sensor. |
| `slides/` | PDF of the IMPC 2026 keynote slides. |

## Quick start

```bash
git clone https://github.com/LidiaAuret/flotation-ml-good-practice.git
cd flotation-ml-good-practice
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # or: pip install -r requirements.txt
cd examples/flotation
python 01_explore_and_splits.py
python 02_evaluation_designs.py
python 03_explanations.py
cd ../e2epm_blending_tank
python get_simulator.py      # clones the E2EPM repository at a pinned commit (needs git)
python simulate.py           # optional: regenerates data/e2epm_blending_tank.csv
python softsensor.py
```

Scripts use `# %%` cells, so they also run interactively in VS Code, Spyder or
Jupyter (via jupytext). Figures go to `figures/`.

## The examples and what they show

### 1. Flotation rougher bank (`examples/flotation`)

Target: plant-style recovery from the two-product formula on assays,
R = c(f - t) / [f(c - t)]. Inputs: feed rate, on-line feed grade, air valves
(cells 1-2, 3-4), frother pump, tail valves, mean bubble size d32, mean
superficial gas velocity jg.

**01 Explore and splits.** Recovery has lag-1 autocorrelation ~0.85 at 1-min
sampling. With n_eff/n = (1 - rho)/(1 + rho), ~20 000 one-minute samples
before the ore change carry the information of only about 1 600 independent
samples, and hourly samples about 100. In a random split, the median distance from a test sample to its
nearest training sample is one minute.

**02 Evaluation designs.** Same data, same models, four test designs:

| Design | Ridge R² | Boosted trees R² |
|---|---|---|
| A Random split of 1-min samples | 0.56 | 0.81 |
| B Blocked time split, 12 h gap, same ore | 0.49 | 0.36 |
| C Train before ore change, test after | 0.11 | 0.01 |
| D As B, plus same-time conc. and tail assays (leaky) | 0.90 | 0.88 |

The random split ranks the boosted trees far ahead; honest designs reverse
the ranking. Design D looks excellent because the target is computed from
two of its inputs (Kapoor & Narayanan's L2, illegitimate features).
The ridge model has 9 parameters; the boosted model ~18 000 tree nodes.

*Fit once, report, deploy.* The second part of 02 fits each model once on
a random 80 % of days 0-10, reports the held-out 20 %, and then applies the
same model, unchanged, to the days that follow:

| Period | Ridge R² | Boosted trees R² |
|---|---|---|
| Random-split test (days 0-10) | 0.49 | 0.78 |
| Later, same ore (days 10-14) | 0.45 | 0.32 |
| Later, after ore change (days 14-21) | 0.04 | -0.06 |

The boosted model's reported 0.78 halves on the very next days; the ridge
model's random-split score holds up far better on the same ore. Neither
survives the new ore. Parity plots: `figures/flotation/02_fit_once.png`.

**03 Explanations.** Standardised ridge coefficients and SHAP slopes are
compared with step-test truth. In closed-loop data, mediators (d32, jg) and
co-moving inputs take credit for each other: neither method recovers the
large negative feed-rate effect, and SHAP credits the air valves with a
small fraction of their real effect. Explanations describe the model, not
the plant. Training the boosted tree on causes only (feed, air, frother;
no tail valves, d32 or jg, which are consequences of those inputs) moves
the SHAP slopes towards the truth: the feed-rate effect grows from -0.05 to
-0.62 against a true -1.76, the air valves rise to about 70 % of their real
effect, and frother gets the right sign. The price is a lower test R²
(0.36 to 0.15), and the feed-grade effect stays overstated. The script also shows that leave-one-out tuning of ridge alpha
(sklearn's `RidgeCV` default) on autocorrelated data picks too little
regularisation and produces unstable coefficients; here alpha is chosen by
forward-chaining CV with a 12 h gap.

### 2. E2EPM blending tank (`examples/e2epm_blending_tank`)

The end-to-end process monitoring framework (Auret & Louw, 2023) simulates a
blending tank with AR(1) feed disturbances, PI concentration control through
a dilution-water valve, a concentration analyser that can drift, a water
valve that can stick open, PCA monitoring and maintenance. `simulate.py`
runs the framework's own Python implementation (100 s step, 28 days) and
records every measurement next to its true value.

`softsensor.py` estimates tank concentration C without the analyser, from
C0, F0, FW, F and level, and compares:

* a training-mean baseline;
* a **zero-parameter mass balance**, dC/dt = [F0(C0 - C) - FW C]/(A L), solved exactly over each sample interval;
* ridge and boosted trees, with and without the mass-balance estimate as a feature (hybrid);
* ridge with the controller output as an extra input.

Findings (RMSE against the true C, kg/m3):

* With true inputs the mass balance reaches R² = 0.88, but the F0 flowmeter
  noise is twice the process variation of F0, so on measured inputs the
  physics estimate is worse than the training mean. Physics structure does
  not rescue an uninformative sensor (Brunton's "data" stage).
* As a feature, the mass balance improves both ridge and boosted trees
  under the blocked split.
* Away from faults, all models are only slightly better than the mean; the
  random split hides this.
* The water valve sticking open drives the process outside the training
  range. The controller output (integral wind-up to -58 against a training
  range of about -0.6 to 0.9) then makes the linear model extrapolate wildly:
  R² of -177 on the blocked test set.
* The analyser drifts during part of the test block, so scoring against the
  analyser (the only label a plant has) and against truth differ.

## Good-practice checklist used in these examples

1. Split in time, with a gap of several process time constants; also test
   out-of-time (new ore, after maintenance).
2. Report the effective sample size, not only the row count.
3. Use only inputs available at prediction time; never the inputs from which
   the target is calculated.
4. Fit preprocessing and tune hyper-parameters inside the training block
   (time-aware CV).
5. Report a simple baseline (mean, persistence, linear) next to the complex
   model.
6. Treat explanations (coefficients, SHAP) as statements about the model;
   test causal claims with step tests or designed experiments.
7. Check labels: an analyser can drift.
8. Share data, code or a simulator so that others can check the result.

## Related resources

* E2EPM framework and ADCHEM 2024 workshop: <https://github.com/LidiaAuret/ADCHEM2024_End-to-end-process-monitoring-workshop>
  (fork of <https://github.com/Stellenbosch-University-Process-Eng/End-to-end-process-monitoring>, MIT licence).
* Kriel, Louw & Auret: open-source crushing, grinding and flotation simulation
  library for end-to-end process monitoring (SAMMRI-sponsored MEng). Link to be
  added when the code is released.
* Kaggle "Quality prediction in a mining process" (iron ore flotation) dataset:
  a useful public plant dataset; note that its assays are forward-filled.

## References

* Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9), 100804.
* Auret, L. & Louw, T.M. (2023). End-to-end process monitoring: Challenges and framework for case study design. *IFAC-PapersOnLine* 56(2), 2650-2656. <https://doi.org/10.1016/j.ifacol.2023.10.1355>
* Auret, L., Haasbroek, A.L., Louw, T.M., Geldenhuys, S. & Bezuidenhout, J. (2025). A dynamic flotation model for rapid prototyping of industrial control and monitoring solutions. *IFAC-PapersOnLine* 59(32), 60-65. <https://doi.org/10.1016/j.ifacol.2025.12.397>
* Bayley, G.V. & Hammersley, J.M. (1946). The "effective" number of independent observations in an autocorrelated time series. *J. R. Stat. Soc. Suppl.* 8(2), 184-197.
* Lundberg, S.M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 56-67.

## Licences and citation

Code: MIT ([`LICENSE`](LICENSE)). Data: CC BY 4.0 ([`data/LICENSE-DATA.md`](data/LICENSE-DATA.md)).
The E2EPM simulator is downloaded separately and keeps its own MIT licence.
Slides: © Stone Three 2026, shared with permission; not covered by the code or data licences.
If you use the data or examples, please cite as in [`CITATION.cff`](CITATION.cff).

To cite the keynote itself:

> Auret, L. (2026). *From promising to proven: Good practice for machine
> learning in flotation* [Keynote address, Flotation session]. XXXII International
> Mineral Processing Congress (IMPC 2026), Cape Town, South Africa, 22 October 2026.
> <https://github.com/LidiaAuret/flotation-ml-good-practice>

Issues and pull requests with further examples (other splits, uncertainty,
physics-informed models) are welcome.

## Contact

Questions, comments and suggestions are welcome: Lidia Auret,
`lidia.auret` *at* `stonethree` *dot* `com`.
