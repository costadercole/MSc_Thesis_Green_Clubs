"""
Goods-market equilibrium with Cournot competition and additive transport costs.

Equations implemented
---------------------
mc_eff      : eq. (3.7)   c_H + t·σ_j  (if H),  c_L  (if L)
c̃_{ji}     : eq. (3.21)  mc_eff + g·1[j≠i] + τ·1[lax→strict] + τ_BA·1[H,lax→strict]
p*_i        : eq. (3.12)  (a + Σ_{j∈A_i} c̃_{ji}) / (|A_i| + 1)   [monotone elim.]
q*_{ji}     : eq. (3.13)  (P_i/b)(p*_i − c̃_{ji})
π_{ji}^var  : eq. (3.16)  (b/P_i)·(q*_{ji})²
π_j         : eq. (3.17)  Σ_i π_{ji}^var − F
"""

import numpy as np
from params import Params


# ---------------------------------------------------------------------------
# Delivered marginal cost — eq. (3.21)
# ---------------------------------------------------------------------------

def _mc_eff(is_H: bool, sigma_j: int, p: Params) -> float:
    """Effective marginal cost at origin (no wage term)."""
    return (p.c_H + p.t * sigma_j) if is_H else p.c_L


def _delivered_cost(is_H: bool, j: int, i: int, sigma: np.ndarray,
                    p: Params) -> float:
    """
    Delivered marginal cost of a firm of type is_H from j into market i.

    Domestic (j == i): mc_eff only.
    Export    (j != i): add additive transport cost g; add τ and τ_BA if lax→strict.
    """
    mc = _mc_eff(is_H, int(sigma[j]), p)
    if j == i:
        return mc
    cost = mc + p.g
    if sigma[j] == 0 and sigma[i] == 1:   # lax origin → strict destination
        cost += p.tau
        if is_H:
            cost += p.tau_BA
    return cost


# ---------------------------------------------------------------------------
# Cournot solver — monotone elimination (§3.4.2)
# ---------------------------------------------------------------------------

def _cournot_solve(
    candidates: list[tuple[float, float]],  # (delivered_cost, firm_count)
    a: float,
) -> tuple[float, list[bool]]:
    """
    Solve Cournot equilibrium via monotone elimination.

    Removes firms with c̃ ≥ p* one pass at a time (each removal weakly lowers p*,
    so the procedure converges monotonically).

    Returns (p_star, active_mask) where active_mask[k] is True iff candidate k
    is in the equilibrium active set.
    """
    if not candidates:
        return a, []

    active = list(range(len(candidates)))
    costs  = [c for c, _ in candidates]
    counts = [n for _, n in candidates]

    while True:
        n_total  = sum(counts[k] for k in active)
        mc_sum   = sum(costs[k] * counts[k] for k in active)
        p        = (a + mc_sum) / (n_total + 1)
        new_act  = [k for k in active if costs[k] < p]
        if len(new_act) == len(active):
            mask = [False] * len(candidates)
            for k in active:
                mask[k] = True
            return p, mask
        if not new_act:
            return a, [False] * len(candidates)
        active = new_act


# ---------------------------------------------------------------------------
# Full market solve — eqs. (3.21), (3.12)–(3.13)
# ---------------------------------------------------------------------------

def solve_market(
    f_H: np.ndarray,    # (N,) H-firm counts per jurisdiction
    f_L: np.ndarray,    # (N,) L-firm counts per jurisdiction
    sigma: np.ndarray,  # (N,) 1=strict, 0=lax
    P: np.ndarray,      # (N,) populations
    W: np.ndarray,      # (N, N) row-stochastic weight matrix (used for neighbour detection only)
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cournot equilibrium with additive transport costs (eqs. 3.21, 3.12–3.13).

    For each destination market i:
      1. Collect domestic firms and neighbors j with W[i,j]>0.
      2. Compute delivered costs c̃_{ji} (additive g, tariff, BCA).
      3. Run monotone elimination → p*_i.
      4. Compute q*_{ji} = (P_i/b)(p*_i − c̃_{ji}) for active (j, type) pairs.

    Returns
    -------
    p_star : (N,)    equilibrium price per jurisdiction.
    q_H    : (N, N)  per-H-firm delivered quantity from j to market i  (q_H[i, j]).
    q_L    : (N, N)  per-L-firm delivered quantity from j to market i  (q_L[i, j]).
    """
    N      = len(f_H)
    p_star = np.full(N, p.a, dtype=float)
    q_H    = np.zeros((N, N), dtype=float)
    q_L    = np.zeros((N, N), dtype=float)

    for i in range(N):
        P_i = max(float(P[i]), 1.0)

        # Build candidate list: (delivered_cost, firm_count, is_H, source_j)
        cands = []

        # Domestic firms (j == i)
        if f_H[i] > 0:
            cands.append((_delivered_cost(True,  i, i, sigma, p), float(f_H[i]), True,  i))
        if f_L[i] > 0:
            cands.append((_delivered_cost(False, i, i, sigma, p), float(f_L[i]), False, i))

        # Neighboring exporters
        for j in range(N):
            if W[i, j] == 0 or i == j:
                continue
            if f_H[j] > 0:
                cands.append((_delivered_cost(True,  j, i, sigma, p), float(f_H[j]), True,  j))
            if f_L[j] > 0:
                cands.append((_delivered_cost(False, j, i, sigma, p), float(f_L[j]), False, j))

        if not cands:
            continue

        mc_count_pairs = [(c, n) for c, n, _, _ in cands]
        p_i, active_mask = _cournot_solve(mc_count_pairs, p.a)
        p_star[i] = p_i

        # Assign quantities for active candidates
        for k, (cost, _, is_H_flag, src) in enumerate(cands):
            if not active_mask[k]:
                continue
            q = (P_i / p.b) * (p_i - cost)
            if is_H_flag:
                q_H[i, src] = max(q, 0.0)
            else:
                q_L[i, src] = max(q, 0.0)

    return p_star, q_H, q_L


# ---------------------------------------------------------------------------
# Firm profits — eq. (3.16)–(3.17)
# ---------------------------------------------------------------------------

def firm_variable_profits(
    q_H: np.ndarray,   # (N, N) per-H-firm delivered quantity from j to i
    q_L: np.ndarray,   # (N, N) per-L-firm delivered quantity from j to i
    P: np.ndarray,     # (N,) populations
    p: Params,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-firm variable profit summed across all active markets (eq. 3.17).

    π_{ji}^var = (b/P_i)·q*_{ji}²

    π_H[j] = Σ_i (b/P_i)·q_H[i,j]²
    π_L[j] = Σ_i (b/P_i)·q_L[i,j]²

    Returns (pi_H, pi_L), each shape (N,).
    """
    P_safe = np.maximum(P, 1.0)
    w = p.b / P_safe          # (N,)

    pi_H = ((q_H ** 2) * w[:, None]).sum(axis=0)
    pi_L = ((q_L ** 2) * w[:, None]).sum(axis=0)

    return pi_H, pi_L


# ---------------------------------------------------------------------------
# Consumer surplus and environmental damage — eqs. (3.20)–(3.21)
# ---------------------------------------------------------------------------

def consumer_surplus(p_star: np.ndarray, a: float, b: float) -> np.ndarray:
    """Per-consumer surplus in each jurisdiction."""
    return (a - p_star) ** 2 / (2 * b)


def environmental_damage(
    f_H: np.ndarray, P: np.ndarray,
    delta_loc: float, delta_glob: float,
) -> np.ndarray:
    """Per-capita environmental damage (local + global components)."""
    P_safe = np.maximum(P, 1.0)
    return delta_loc * f_H / P_safe + delta_glob * f_H.sum() / P_safe.sum()
