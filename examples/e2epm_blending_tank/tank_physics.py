"""Zero-parameter mass-balance soft sensor for the blending tank.

Tank (cross-section A = 4 m2) with feed F0 at concentration C0 and dilution
water FW. Assuming perfect mixing,

    dV/dt  = F0 + FW - F,          V = A L
    d(CV)/dt = C0 F0 - C F
 => dC/dt  = [F0 (C0 - C) - FW C] / V

Holding the measured inputs constant over a sample interval dt (zero-order
hold), the exact solution is a first-order step towards the instantaneous
steady state C_ss = F0 C0 / (F0 + FW) with time constant tau = V / (F0 + FW):

    C[k+1] = C_ss + (C[k] - C_ss) exp(-dt / tau)

No parameter is fitted to data: the only constant, A, comes from the design.
"""
from __future__ import annotations

import numpy as np

AREA_M2 = 4.0


def mass_balance_estimate(F0, C0, FW, L, running, C_init, dt_s: float = 100.0,
                          area: float = AREA_M2, min_level: float = 0.05) -> np.ndarray:
    """Run the mass balance forward on measured inputs.

    The state is re-initialised from `C_init` (e.g. the analyser) whenever the
    tank is not in the running regime, so each running period starts from a
    known value and is then predicted open-loop, without the analyser.
    """
    F0, C0, FW, L, C_init = (np.asarray(a, float) for a in (F0, C0, FW, L, C_init))
    running = np.asarray(running, bool)
    out = np.empty(len(F0))
    out[0] = C_init[0]
    for i in range(1, len(F0)):
        if not (running[i] and running[i - 1]):
            out[i] = C_init[i]
            continue
        f0, fw = max(F0[i - 1], 0.0), max(FW[i - 1], 0.0)
        q = max(f0 + fw, 1e-9)
        c_ss = f0 * C0[i - 1] / q
        tau = area * max(L[i - 1], min_level) / q
        out[i] = c_ss + (out[i - 1] - c_ss) * np.exp(-dt_s / tau)
    return out
