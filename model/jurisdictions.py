"""
Jurisdiction state and dynamics.

State
-----
sigma : (N,) int  — regulatory policy: 1 = strict (S), 0 = lax (L)

Dynamics
--------
- Per-capita welfare (eq. 3.22)
- Tariff payoffs (eq. 3.6)
- 2×2 payoff matrix A (eqs. 3.24–3.27)
- Network correction B (eqs. 3.29–3.30)
- Policy replicator ṡ (eqs. 3.31–3.33)
"""

import numpy as np
from params import Params


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_jurisdictions(p: Params, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign regulatory policies so that fraction s0 are strict."""
    sigma = (rng.random(p.N) < p.s0).astype(int)
    return sigma


def init_populations(p: Params, rng: np.random.Generator) -> np.ndarray:
    """Draw jurisdiction populations from log-normal (eq. 3.1)."""
    log_P = rng.normal(p.mu_P, p.sigma_P, size=p.N)
    return np.exp(log_P)


# ---------------------------------------------------------------------------
# Tariff payoffs — eq. (3.6)
# ---------------------------------------------------------------------------

def tariff_payoffs(
    sigma: np.ndarray,  # (N,)
    W: np.ndarray,      # (N, N) weight matrix
    tau: float,
) -> np.ndarray:
    """
    Net tariff payoff T_i for each jurisdiction (eq. 3.6).

    Strict i collects tau * sum_j w_ij * 1[sigma_j = L].
    Lax   i pays    tau * sum_j w_ij * 1[sigma_j = S].
    """
    lax = (sigma == 0).astype(float)
    strict = (sigma == 1).astype(float)
    T = np.where(
        sigma == 1,
         tau * (W @ lax),
        -tau * (W @ strict),
    )
    return T


# ---------------------------------------------------------------------------
# Per-capita welfare — eq. (3.22)
# ---------------------------------------------------------------------------

def per_capita_welfare(
    f_H: np.ndarray,    # (N,)
    f_L: np.ndarray,    # (N,)
    sigma: np.ndarray,  # (N,)
    P: np.ndarray,      # (N,)
    p_star: np.ndarray, # (N,)
    W: np.ndarray,      # (N, N)
    p: Params,
) -> np.ndarray:
    """
    Per-capita welfare W_i / P_i (eq. 3.22).

    = w_bar * f_i/P_i  +  (a - p*_i)^2 / (2b)  +  TR_i/P_i  -  delta * f_H_i/P_i
    """
    f_i = f_H + f_L
    wages = p.w_bar * f_i / np.maximum(P, 1)
    cs = (p.a - p_star) ** 2 / (2 * p.b)

    T = tariff_payoffs(sigma, W, p.tau)
    tax_revenue = p.t * f_H * sigma   # total carbon tax receipts (eq. 3.22)
    TR = tax_revenue + T              # total fiscal revenue = tax + net tariff (eq. 3.22)
    fiscal = TR / np.maximum(P, 1)

    damage = p.delta * f_H / np.maximum(P, 1)

    return wages + cs + fiscal - damage


# ---------------------------------------------------------------------------
# Payoff matrix — eqs. (3.23)–(3.27)
# ---------------------------------------------------------------------------

def payoff_matrix(
    f_H: np.ndarray,
    f_L: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    p_star: np.ndarray,
    W: np.ndarray,
    p: Params,
) -> tuple[float, float, float, float]:
    """
    Compute the 2×2 payoff matrix entries (a_SS, a_SL, a_LS, a_LL).

    Uses mean quantities over strict / lax jurisdictions as defined in §3.8.1.
    Returns (a_SS, a_SL, a_LS, a_LL).
    """
    strict_mask = sigma == 1
    lax_mask = sigma == 0

    def _mean(arr, mask):
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    f_i = f_H + f_L

    # Strict averages
    fS_bar  = _mean(f_i, strict_mask)
    fHS_bar = _mean(f_H, strict_mask)
    PS_bar  = _mean(P, strict_mask)
    pS_bar  = _mean(p_star, strict_mask)

    # Lax averages
    fL_bar  = _mean(f_i, lax_mask)
    fHL_bar = _mean(f_H, lax_mask)
    PL_bar  = _mean(P, lax_mask)
    pL_bar  = _mean(p_star, lax_mask)

    inv_PS = 1.0 / max(PS_bar, 1e-9)
    inv_PL = 1.0 / max(PL_bar, 1e-9)

    # Abatement / admin cost κ is not separately modelled; set to 0.
    kappa = 0.0

    a_SS = (p.w_bar * fS_bar * inv_PS
            + (p.a - pS_bar) ** 2 / (2 * p.b)
            + p.t * fHS_bar * inv_PS
            - kappa * inv_PS
            - p.delta * fHS_bar * inv_PS)

    a_SL = (p.w_bar * fS_bar * inv_PS
            + (p.a - pS_bar) ** 2 / (2 * p.b)
            + p.t * fHS_bar * inv_PS
            + p.tau * inv_PS
            - kappa * inv_PS
            - p.delta * fHS_bar * inv_PS)

    a_LS = (p.w_bar * fL_bar * inv_PL
            + (p.a - pL_bar) ** 2 / (2 * p.b)
            - p.tau * inv_PL
            - p.delta * fHL_bar * inv_PL)

    a_LL = (p.w_bar * fL_bar * inv_PL
            + (p.a - pL_bar) ** 2 / (2 * p.b)
            - p.delta * fHL_bar * inv_PL)

    return a_SS, a_SL, a_LS, a_LL


# ---------------------------------------------------------------------------
# Network correction — eqs. (3.29)–(3.30)
# ---------------------------------------------------------------------------

def network_correction(
    a_SS: float, a_SL: float, a_LS: float, a_LL: float,
    k: float,
) -> float:
    """
    Compute b_SL, the single non-trivial entry of the correction matrix B (eq. 3.30).

    b_SS = b_LL = 0  (antisymmetry from [7]).
    Requires k > 2.
    """
    if k <= 2:
        return 0.0
    b_SL = ((k + 3) * a_SS + 3 * a_SL - 3 * a_LS - (k + 3) * a_LL) / ((k + 3) * (k - 2))
    return float(b_SL)


# ---------------------------------------------------------------------------
# Policy replicator — eqs. (3.31)–(3.33)
# ---------------------------------------------------------------------------

def policy_replicator(
    s: float,
    a_SS: float, a_SL: float, a_LS: float, a_LL: float,
    b_SL: float,
    dt: float,
) -> float:
    """
    Euler step for s (fraction of strict jurisdictions), eq. (3.31).

    ṡ = s(1-s)(Pi_tilde_S - Pi_tilde_L)
    """
    a_tilde_SS = a_SS          # b_SS = 0
    a_tilde_SL = a_SL + b_SL
    a_tilde_LS = a_LS - b_SL   # b_LS = -b_SL
    a_tilde_LL = a_LL          # b_LL = 0

    Pi_S = s * a_tilde_SS + (1 - s) * a_tilde_SL
    Pi_L = s * a_tilde_LS + (1 - s) * a_tilde_LL

    s_dot = s * (1 - s) * (Pi_S - Pi_L)
    return float(np.clip(s + dt * s_dot, 0.0, 1.0))
