"""Generate the E2EPM blending-tank dataset used by softsensor.py.

    python get_simulator.py      # once
    python simulate.py           # ~5 s; writes ../../data/e2epm_blending_tank.csv

Runs the Python (lower sampling frequency, 100 s step) implementation of the
end-to-end process monitoring framework (Auret & Louw, 2023): a blending tank
with AR(1) feed disturbances, PI concentration control via the water valve,
a drifting concentration sensor, a valve that can stick open, PCA monitoring,
and planned/unplanned maintenance. Every sample records what the operator can
see (measured values) next to the simulator's true values.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE / "external" / "ADCHEM2024_E2EPM" / "Python_lower_sampling_frequency"
OUT = HERE.parents[1] / "data" / "e2epm_blending_tank.csv"


def run(seed: int = 85, maintenance_strategy: int = 2) -> pd.DataFrame:
    if not SIM.exists():
        sys.exit("Simulator not found: run `python get_simulator.py` first.")
    sys.path.insert(0, str(SIM))
    from simulation_configs import generateConfigurations
    from simulation_modules import (
        DisturbanceModule, EconomicModule, FaultModule, MaintenanceModule, MeasurementModule,
        MonitoringModule, ProcessModule, RegulatoryControlModule, SupervisoryControlModule,
    )

    np.random.seed(seed)
    t, d, u, p, x, f, y, r, m, econ = generateConfigurations(maintenance_strategy=maintenance_strategy)
    d_range = DisturbanceModule(d, t)
    rows = []
    # Same loop as BlendingTank_IFACWC.py in the workshop repository, with a richer record.
    while t["tvector"][t["i"]] < t["tmax"]:
        t["tvector"][t["i"] + 1] = t["tvector"][t["i"]] + t["deltat"]
        d_now = {c: d_range[c][t["i"]] for c in ["F0", "C0"]}
        r = SupervisoryControlModule(r, m, y, t)
        if r["regime"] == "Shut":
            r, f, t = MaintenanceModule(r, f, t)
        u = RegulatoryControlModule(u, y, r)
        x = ProcessModule(p, u, d_now, x, f, t)
        f = FaultModule(p, x, f, t)
        y = MeasurementModule(y, x, d_now, f)
        m = MonitoringModule(m, y, r, t)
        econ = EconomicModule(econ, x, r, t)
        rows.append({
            "time_h": t["tvector"][t["i"]] / 3600,
            "regime": r["regime"],
            # measured (what the operator and any model can use)
            "C_meas": y["C"]["value"], "C0_meas": y["C0"]["value"], "F0_meas": y["F0"]["value"],
            "FW_meas": y["FW"]["value"], "F_meas": y["F"]["value"], "L_meas": y["L"]["value"],
            "water_valve_command": u["xWv"],
            # truth (simulation only)
            "truth_C": x["C"], "truth_C0": d_now["C0"], "truth_F0": x["F0"], "truth_FW": x["FW"],
            "truth_F": x["F"], "truth_L": x["L"], "truth_water_valve": x["xWv"],
            "fault_C_sensor_drift": float(f["C"]["state"] != "none"),
            "fault_C_sensor_offset": f["C"].get("drift", 0.0),
            "fault_water_valve_stuck": float(f["valveFW"]["state"] != "none"),
        })
        t["i"] += 1
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=85)
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")  # the monitoring module emits harmless ComplexWarnings
    df = run(args.seed)
    OUT.parent.mkdir(exist_ok=True)
    df.round(6).to_csv(OUT, index=False)
    print(f"{len(df)} samples, {df['time_h'].max() / 24:.1f} days -> {OUT}")
    print(df["regime"].value_counts().to_string())


if __name__ == "__main__":
    main()
