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


def classify_outcome(h_ss: float, s_ss: float) -> str:
    """
    Label the steady-state outcome using a 3×3 grid in (h, s) space.

    h axis (high-emission firm share):
      low    h < 0.33   — mostly clean firms
      mid    0.33–0.67  — mixed firm composition
      high   h > 0.67   — mostly dirty firms

    s axis (strict jurisdiction share):
      low    s < 0.33   — mostly lax jurisdictions
      mid    0.33–0.67  — mixed policy
      high   s > 0.67   — mostly strict jurisdictions

    Labels (9 cells):
      green_club          h_low,  s_high  — the target outcome
      partial_club        h_low,  s_mid   — some clubs, mostly clean
      clean_lax           h_low,  s_low   — clean firms but no policy
      dirty_strict        h_high, s_high  — strict policy but leaky
      race_to_bottom      h_high, s_low   — dirty firms, no policy
      dirty_mixed_policy  h_high, s_mid
      transitional_clean  h_mid,  s_high
      transitional        h_mid,  s_mid
      transitional_lax    h_mid,  s_low
    """
    h_low  = h_ss < 0.33
    h_high = h_ss > 0.67
    s_low  = s_ss < 0.33
    s_high = s_ss > 0.67

    if h_low  and s_high: return "green_club"
    if h_low  and s_low:  return "clean_lax"
    if h_low:             return "partial_club"
    if h_high and s_low:  return "race_to_bottom"
    if h_high and s_high: return "dirty_strict"
    if h_high:            return "dirty_mixed_policy"
    if s_high:            return "transitional_clean"
    if s_low:             return "transitional_lax"
    return "transitional"
