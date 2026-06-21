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
- Firm location: H-firms relocate to neighbouring jurisdictions via eq. (3.36).
  Disabled when p.relocate is False.
- Emission type: agent-level Fermi imitation (eq. 3.42).  Each firm with
  probability nu*dt samples a random partner from the global pool and adopts
  its type with Fermi probability expit(kappa_f * (profit_partner - profit_self)).
  Rate nu sets the timescale; kappa_f sets sharpness.  Mean-field limit recovers
  the standard replicator h_dot = nu*h*(1-h)*tanh(kappa_f*(pi_H_bar - pi_L_bar)/2).
"""

import numpy as np
from scipy.special import expit
from params import Params


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_firms(p: Params, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Randomly assign M firms to jurisdictions and emission types."""
    firm_loc  = rng.integers(0, p.N, size=p.M)
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
# Relocation dynamics — eq. (3.36)
# ---------------------------------------------------------------------------

def relocate_firms(
    firm_loc: np.ndarray,
    firm_type: np.ndarray,
    sigma: np.ndarray,       # (N,) regulatory policy: 1=S, 0=L
    pi_H: np.ndarray,        # (N,) per-firm variable profit of H-type, per jurisdiction
    p: Params,
    rng: np.random.Generator,
    W: np.ndarray = None,    # (N, N) weight matrix; restricts relocation to neighbours
) -> np.ndarray:
    """
    Update firm_loc for high-emission firms based on profit incentives (eq. 3.36).

    For each H-firm in jurisdiction i, find the most profitable neighbouring
    jurisdiction j.  If pi_H[j] > pi_H[i], move with probability
    mu * min(1, delta_pi/pi_ref) * dt.
    """
    if not p.relocate:
        return firm_loc

    h_firms = np.where(firm_type == 1)[0]
    if len(h_firms) == 0:
        return firm_loc

    lax_mask = sigma == 0
    if lax_mask.any() and pi_H[lax_mask].max() > 0:
        pi_ref = float(np.mean(pi_H[lax_mask]))
    else:
        pi_ref = float(np.mean(pi_H[pi_H > 0])) if (pi_H > 0).any() else 1e-9
    pi_ref = max(pi_ref, 1e-9)

    for idx in h_firms:
        i = firm_loc[idx]

        if W is not None:
            candidates = [j for j in range(len(sigma)) if W[i, j] > 0]
        else:
            candidates = [j for j in range(len(sigma)) if j != i]

        if not candidates:
            continue

        best_dest = candidates[int(np.argmax(pi_H[candidates]))]
        delta_pi  = pi_H[best_dest] - pi_H[i]
        mu_ij     = p.mu * min(1.0, max(0.0, delta_pi / pi_ref))
        if mu_ij > 0 and rng.random() < mu_ij * p.dt:
            firm_loc[idx] = best_dest

    return firm_loc


# ---------------------------------------------------------------------------
# Emission type update — agent-level Fermi imitation (eq. 3.42)
# ---------------------------------------------------------------------------

def firm_type_update(
    firm_type: np.ndarray,   # (M,) int  1=H, 0=L
    profit: np.ndarray,      # (M,) float  realised profit of each firm this step
    nu: float,               # revision rate  (timescale knob, same units as lambda/mu)
    kappa_f: float,          # selection intensity  (sharpness knob)
    dt: float,
    rng: np.random.Generator,
    eps: float = 0.0,        # spontaneous mutation rate (keeps boundaries leaky)
) -> np.ndarray:
    """
    Each firm independently gets a revision opportunity with prob nu*dt.
    On revision: sample one partner uniformly from the GLOBAL pool (well-mixed),
    adopt partner's type with Fermi probability expit(kappa_f * (pi_partner - pi_self)).

    Returns a new array (synchronous: reads from old firm_type, writes to copy).

    Mean-field limit: h_dot = nu * h*(1-h) * tanh(kappa_f * (pi_H_bar - pi_L_bar) / 2)
    Rate is bounded by nu regardless of profit magnitude — that is the fix.
    """
    M = firm_type.size
    new_type = firm_type.copy()

    revise = rng.random(M) < nu * dt
    idx = np.where(revise)[0]

    if idx.size > 0:
        partners = rng.integers(0, M, size=idx.size)
        # forbid self-sampling (would be a no-op but skews the h(1-h) weighting)
        clash = partners == idx
        while clash.any():
            partners[clash] = rng.integers(0, M, size=int(clash.sum()))
            clash = partners == idx

        dpi = profit[partners] - profit[idx]
        p_switch = expit(kappa_f * dpi)   # scipy.special.expit: stable for large |x|
        do_switch = rng.random(idx.size) < p_switch
        # read partner types from OLD array (synchronous update)
        new_type[idx[do_switch]] = firm_type[partners[do_switch]]

    if eps > 0.0:
        flip = rng.random(M) < eps * dt
        new_type[flip] = 1 - new_type[flip]

    return new_type
