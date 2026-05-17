"""
Firm-level state and dynamics.

State
-----
firm_loc  : (M,) int   — jurisdiction index for each firm
firm_type : (M,) int   — 1 = high-emission (H), 0 = low-emission (L)

Derived count arrays (recomputed each step from the flat arrays):
f_H[i], f_L[i] — number of H- and L-type firms in jurisdiction i.

Dynamics
--------
- Firm location: high-emission firms in strict jurisdictions may relocate to lax
  jurisdictions via eq. (3.17).
- Emission strategy share h: replicator equation (3.18).
"""

import numpy as np
from params import Params


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_firms(p: Params, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly assign M firms to jurisdictions and emission types.

    Returns (firm_loc, firm_type).
    """
    firm_loc = rng.integers(0, p.N, size=p.M)
    firm_type = (rng.random(p.M) < p.h0).astype(int)   # 1 = H, 0 = L
    return firm_loc, firm_type


# ---------------------------------------------------------------------------
# Count arrays
# ---------------------------------------------------------------------------

def count_firms(
    firm_loc: np.ndarray,
    firm_type: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive f_H (N,) and f_L (N,) from flat firm arrays."""
    f_H = np.bincount(firm_loc[firm_type == 1], minlength=N).astype(float)
    f_L = np.bincount(firm_loc[firm_type == 0], minlength=N).astype(float)
    return f_H, f_L


# ---------------------------------------------------------------------------
# Relocation dynamics — eq. (3.17)
# ---------------------------------------------------------------------------

def relocate_firms(
    firm_loc: np.ndarray,
    firm_type: np.ndarray,
    sigma: np.ndarray,       # (N,) regulatory policy: 1=S, 0=L
    pi_H: np.ndarray,        # (N,) per-firm variable profit of H-type, per jurisdiction
    p: Params,
    rng: np.random.Generator,
    W: np.ndarray = None,    # (N, N) weight matrix; if provided, restricts relocation to neighbours
) -> np.ndarray:
    """
    Update firm_loc for high-emission firms based on profit incentives (eq. 3.17).

    For each H-firm in jurisdiction i, find the most profitable neighbouring
    jurisdiction j (any policy type).  If pi_H[j] > pi_H[i], compute the
    relocation probability µ_ij = µ · min(1, Δπ/π_ref) and move with
    probability µ_ij · dt.  Leakage (strict→lax moves) emerges from profit
    incentives rather than being hard-coded by policy type.

    Restricts to W-neighbours to prevent unrealistic global teleportation.
    Falls back to all jurisdictions if W is None.

    Returns updated firm_loc.
    """
    h_firms = np.where(firm_type == 1)[0]
    if len(h_firms) == 0:
        return firm_loc

    for idx in h_firms:
        i = firm_loc[idx]

        # Candidate destinations: all neighbours of jurisdiction i
        if W is not None:
            candidates = [j for j in range(len(sigma)) if W[i, j] > 0]
        else:
            candidates = [j for j in range(len(sigma)) if j != i]

        if not candidates:
            continue

        best_dest = candidates[int(np.argmax(pi_H[candidates]))]
        delta_pi  = pi_H[best_dest] - pi_H[i]
        pi_origin = max(float(pi_H[i]), 1e-9)   # profit at origin = normaliser
        mu_ij = p.mu * min(1.0, max(0.0, delta_pi / pi_origin))
        if mu_ij > 0 and rng.random() < mu_ij * p.dt:
            firm_loc[idx] = best_dest

    return firm_loc


# ---------------------------------------------------------------------------
# Emission strategy replicator — eq. (3.18)
# ---------------------------------------------------------------------------

def emission_replicator(
    h: float,
    pi_H_bar: float,   # average profit of H-firms across all jurisdictions
    pi_L_bar: float,   # average profit of L-firms across all jurisdictions
    dt: float,
) -> float:
    """
    Euler step for h (share of high-emission firms).

    h_dot = h (1-h) (pi_H_bar - pi_L_bar)
    """
    h_dot = h * (1 - h) * (pi_H_bar - pi_L_bar)
    return float(np.clip(h + dt * h_dot, 0.0, 1.0))


def average_profits(
    firm_loc: np.ndarray,
    firm_type: np.ndarray,
    pi_H: np.ndarray,   # (N,) variable profit per H-firm in each jurisdiction
    pi_L: np.ndarray,   # (N,) variable profit per L-firm in each jurisdiction
    F: float,
) -> tuple[float, float]:
    """
    Compute pi_H_bar and pi_L_bar across all active firms (eq. 3.14).

    Variable profit per firm is taken from the jurisdiction the firm resides in.
    Fixed cost F is subtracted to give total profit.
    """
    h_mask = firm_type == 1
    l_mask = firm_type == 0

    pi_H_bar = float(np.mean(pi_H[firm_loc[h_mask]] - F)) if h_mask.any() else 0.0
    pi_L_bar = float(np.mean(pi_L[firm_loc[l_mask]] - F)) if l_mask.any() else 0.0
    return pi_H_bar, pi_L_bar
