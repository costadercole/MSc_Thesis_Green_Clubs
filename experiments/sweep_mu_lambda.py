"""
Sweep µ/λ to locate the critical threshold (µ/λ)* at which the system
transitions between the green-club and race-to-the-bottom equilibria.
"""

import numpy as np
from dataclasses import replace
from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state, classify_outcome


def sweep(
    mu_lam_values: np.ndarray | None = None,
    lam: float = 0.5,
) -> list[dict]:
    if mu_lam_values is None:
        mu_lam_values = np.linspace(0.1, 5.0, 30)

    results = []
    for ratio in mu_lam_values:
        p = replace(BASELINE, mu=ratio * lam, lam=lam)
        res = run(p)
        ss = steady_state(res)
        outcome = classify_outcome(ss["h_ss"], ss["s_ss"])
        results.append({"mu_lam": ratio, **ss, "outcome": outcome})
        print(f"  µ/λ={ratio:.2f}  →  h_ss={ss['h_ss']:.3f}  s_ss={ss['s_ss']:.3f}  [{outcome}]")

    return results


if __name__ == "__main__":
    rows = sweep()
