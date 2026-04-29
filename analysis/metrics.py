"""
Summary statistics computed from a simulation results dict.
"""

import numpy as np


def steady_state(results: dict, tail: float = 0.2) -> dict:
    """
    Average h and s over the last `tail` fraction of time steps.

    Returns a dict with keys: h_ss, s_ss, h_std, s_std.
    """
    T = len(results["h"])
    start = int(T * (1 - tail))
    return {
        "h_ss":  float(np.mean(results["h"][start:])),
        "s_ss":  float(np.mean(results["s"][start:])),
        "h_std": float(np.std(results["h"][start:])),
        "s_std": float(np.std(results["s"][start:])),
    }


def convergence_time(series: np.ndarray, threshold: float = 0.05) -> int:
    """
    First step at which |x - x_final| < threshold permanently.

    Returns -1 if the series never converges.
    """
    x_final = series[-1]
    for t in range(len(series) - 1, -1, -1):
        if abs(series[t] - x_final) >= threshold:
            return t + 1
    return 0


def classify_outcome(h_ss: float, s_ss: float, tol: float = 0.1) -> str:
    """
    Label the steady-state outcome.

    green_club        : h ≈ 0, s ≈ 1
    race_to_bottom    : h ≈ 1, s ≈ 0
    mixed             : neither corner
    full_dirty_strict : h ≈ 1, s ≈ 1  (strict but with dirty firms)
    full_clean_lax    : h ≈ 0, s ≈ 0
    """
    h0 = h_ss < tol
    h1 = h_ss > 1 - tol
    s0 = s_ss < tol
    s1 = s_ss > 1 - tol

    if h0 and s1:
        return "green_club"
    if h1 and s0:
        return "race_to_bottom"
    if h0 and s0:
        return "full_clean_lax"
    if h1 and s1:
        return "full_dirty_strict"
    return "mixed"
