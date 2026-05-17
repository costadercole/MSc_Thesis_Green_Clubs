"""
Labour market — §3.7.

Each jurisdiction i has fixed labour supply L_i^s = P_i.
Labour demand aggregates the iceberg-adjusted production of all firms based in i:

    L_i^d = (1/α) · Σ_{j: l_j=i} Σ_m q*_{jm} / w_{mi}

where q*_{jm} is the delivered quantity per firm from i to market m, and
w_{mi} = W[m, i] for m≠i (iceberg weight), w_{ii} = 1 (no melt domestically).

In the slack-free regime the wage w_i adjusts so L_i^d(w_i) = P_i.
Below the reservation wage w̄ the market is slack and w_i = w̄.

The outer fixed-point loop (wage → market → labour demand → wage) is solved by
damped proportional updating, which converges in ≈10–20 iterations.
"""

import numpy as np
from params import Params
from model.market import solve_market


# ---------------------------------------------------------------------------
# Labour demand — eq. (3.24)
# ---------------------------------------------------------------------------

def labour_demand(
    f_H: np.ndarray,   # (N,)
    f_L: np.ndarray,   # (N,)
    q_H: np.ndarray,   # (N, N) per-H-firm delivered qty from j to i  (q_H[i, j])
    q_L: np.ndarray,   # (N, N) per-L-firm delivered qty from j to i  (q_L[i, j])
    W: np.ndarray,     # (N, N)
    p: Params,
) -> np.ndarray:
    """
    Labour demand in each jurisdiction (eq. 3.24).

    L_i^d = (1/α) · [f_H[i] · Σ_m q_H[m,i]/w_{mi}  +  f_L[i] · Σ_m q_L[m,i]/w_{mi}]

    where w_{ii} = 1 (domestic, no iceberg melt) and w_{mi} = W[m, i] for m≠i.
    """
    N = len(f_H)
    L_d = np.zeros(N)

    for i in range(N):
        prod_H = float(q_H[i, i])   # domestic: produced = delivered (w=1)
        prod_L = float(q_L[i, i])

        for m in range(N):
            if m == i or W[m, i] == 0:
                continue
            prod_H += float(q_H[m, i]) / float(W[m, i])   # export: divide by iceberg
            prod_L += float(q_L[m, i]) / float(W[m, i])

        L_d[i] = (f_H[i] * prod_H + f_L[i] * prod_L) / p.alpha

    return L_d


# ---------------------------------------------------------------------------
# Wage fixed-point solver — eq. (3.25)
# ---------------------------------------------------------------------------

def solve_wages(
    f_H: np.ndarray,
    f_L: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    W: np.ndarray,
    p: Params,
    max_iter: int = 20,
    tol: float = 1e-4,
    damp: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the wage fixed point L_i^d(w) = P_i via damped iteration.

    Algorithm
    ---------
    1. Start at reservation wage w̄.
    2. Solve goods market → (p*, q_H, q_L).
    3. Compute labour demand L_d.
    4. Update:  w_i ← max(w̄,  w_i · (L_d[i]/P_i)^damp)
    5. Repeat until max|Δw| < tol or max_iter reached.

    A higher ratio L_d/P → excess demand → raise w → higher delivered costs →
    less production → L_d falls back toward P.  Monotone in w_i.

    Returns
    -------
    wages  : (N,) converged wages
    p_star : (N,) equilibrium prices at converged wages
    q_H    : (N, N) per-H-firm quantities
    q_L    : (N, N) per-L-firm quantities
    """
    N      = len(f_H)
    wages  = np.full(N, p.w_bar, dtype=float)
    P_safe = np.maximum(P, 1.0)

    p_star = np.full(N, p.a, dtype=float)
    q_H    = np.zeros((N, N), dtype=float)
    q_L    = np.zeros((N, N), dtype=float)

    for _ in range(max_iter):
        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, wages, p)
        L_d = labour_demand(f_H, f_L, q_H, q_L, W, p)

        ratio     = L_d / P_safe
        wages_new = np.where(ratio > 1.0,
                             wages * (ratio ** damp),
                             p.w_bar)
        wages_new = np.maximum(wages_new, p.w_bar)

        if np.max(np.abs(wages_new - wages)) < tol:
            wages = wages_new
            break
        wages = wages_new

    # Final market solve at converged wages
    p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, wages, p)
    return wages, p_star, q_H, q_L
