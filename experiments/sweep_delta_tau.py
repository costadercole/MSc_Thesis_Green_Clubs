"""
Joint sweep over the damage parameter δ and tariff rate τ.

Produces a 2-D grid of steady-state outcomes (h_ss, s_ss).
"""

import numpy as np
from dataclasses import replace
from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state, classify_outcome


def sweep(
    delta_values: np.ndarray | None = None,
    tau_values: np.ndarray | None = None,
) -> dict:
    if delta_values is None:
        delta_values = np.linspace(0.0, 5.0, 10)
    if tau_values is None:
        tau_values = np.linspace(0.0, 3.0, 10)

    h_grid = np.empty((len(tau_values), len(delta_values)))
    s_grid = np.empty((len(tau_values), len(delta_values)))

    for j, delta in enumerate(delta_values):
        for i, tau in enumerate(tau_values):
            p = replace(BASELINE, delta=delta, tau=tau)
            res = run(p)
            ss = steady_state(res)
            h_grid[i, j] = ss["h_ss"]
            s_grid[i, j] = ss["s_ss"]
            print(f"  δ={delta:.2f}  τ={tau:.2f}  →  h={ss['h_ss']:.3f}  s={ss['s_ss']:.3f}")

    return {
        "delta_values": delta_values,
        "tau_values":   tau_values,
        "h_grid":       h_grid,
        "s_grid":       s_grid,
    }


if __name__ == "__main__":
    sweep()
