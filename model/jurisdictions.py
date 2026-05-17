"""
Jurisdiction state and dynamics.

State
-----
sigma : (N,) int  — regulatory policy: 1 = strict (S), 0 = lax (L)

Fiscal revenues (eqs. 3.22–3.24)
---------------------------------
TR_i^tax     = t·1[S]·f_H[i]·Σ_m q_H[m,i]/w_{mi}          (carbon tax on total production)
TR_i^tariff  = τ·1[S]·Σ_{ℓ lax nbr} (f_H[ℓ]·q_H[i,ℓ] + f_L[ℓ]·q_L[i,ℓ])
TR_i^BCA     = τ_BA·1[S]·Σ_{ℓ lax nbr} f_H[ℓ]·q_H[i,ℓ]

Per-capita welfare (eq. 3.26 — slack-free):
W_i/P_i = w_i + (a−p*_i)²/(2b) + TR_i/P_i − δ·f_H[i]/P_i

Payoff matrix (eqs. 3.28–3.31):
a_SS = W_S^base + t·Q̄^prod_{H,S}/P̄_S
a_SL = a_SS + τ·q̄^imp_{S←L}/P̄_S + τ_BA·q̄^{H,imp}_{S←L}/P̄_S
a_LS = a_LL = W_L^base
where W_X^base = w̄_X + cs_X − δ·f̄^H_X/P̄_X

Policy dynamics (eqs. 3.32–3.34):
ṡ = s(1−s)(Π̃_S − Π̃_L)  with network correction B (Ohtsuki 2006).
"""

import numpy as np
from params import Params


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_jurisdictions(p: Params, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(p.N) < p.s0).astype(int)


def init_populations(p: Params, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.normal(p.mu_P, p.sigma_P, size=p.N))


# ---------------------------------------------------------------------------
# Fiscal revenues — eqs. (3.22)–(3.24)
# ---------------------------------------------------------------------------

def fiscal_revenues(
    f_H: np.ndarray,    # (N,)
    f_L: np.ndarray,    # (N,)
    sigma: np.ndarray,  # (N,)
    W: np.ndarray,      # (N, N)
    q_H: np.ndarray,    # (N, N) per-H-firm delivered qty from j to i
    q_L: np.ndarray,    # (N, N) per-L-firm delivered qty from j to i
    p: Params,
) -> np.ndarray:
    """
    Total fiscal revenue TR_i for each jurisdiction (eq. 3.24).

    Strict jurisdictions collect carbon-tax, tariff, and BCA revenue.
    Lax jurisdictions earn zero direct fiscal revenue.
    Returns TR (N,).
    """
    N  = len(f_H)
    TR = np.zeros(N)

    for i in range(N):
        if sigma[i] == 0:
            continue   # lax: no direct fiscal revenue

        # ── Carbon-tax revenue (eq. 3.22) ───────────────────────────────────
        # t · f_H[i] · (domestic production + export production / iceberg weight)
        prod_H_i = float(q_H[i, i])                     # domestic, w=1
        for m in range(N):
            if m != i and W[m, i] > 0:
                prod_H_i += float(q_H[m, i]) / float(W[m, i])   # iceberg melt
        TR[i] += p.t * f_H[i] * prod_H_i

        # ── Tariff + BCA revenue (eqs. 3.23–3.24) ───────────────────────────
        for ell in range(N):
            if W[i, ell] == 0 or ell == i:
                continue
            if sigma[ell] != 0:           # tariff only from lax origins
                continue
            q_tot_imp = f_H[ell] * float(q_H[i, ell]) + f_L[ell] * float(q_L[i, ell])
            q_H_imp   = f_H[ell] * float(q_H[i, ell])
            TR[i] += p.tau    * q_tot_imp
            TR[i] += p.tau_BA * q_H_imp

    return TR


# ---------------------------------------------------------------------------
# Per-capita welfare — eq. (3.26)
# ---------------------------------------------------------------------------

def per_capita_welfare(
    f_H: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    p_star: np.ndarray,
    wages: np.ndarray,  # (N,) endogenous wages
    TR: np.ndarray,     # (N,) total fiscal revenue from fiscal_revenues()
    p: Params,
) -> np.ndarray:
    """
    Per-capita welfare W_i/P_i (eq. 3.26, slack-free form).

    = w_i  +  (a−p*_i)²/(2b)  +  TR_i/P_i  −  δ·f_H[i]/P_i
    """
    P_safe = np.maximum(P, 1.0)
    cs     = (p.a - p_star) ** 2 / (2 * p.b)
    damage = p.delta * f_H / P_safe
    fiscal = TR / P_safe
    return wages + cs + fiscal - damage


# ---------------------------------------------------------------------------
# Payoff matrix — eqs. (3.28)–(3.31)
# ---------------------------------------------------------------------------

def payoff_matrix(
    f_H: np.ndarray,
    f_L: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    p_star: np.ndarray,
    W: np.ndarray,
    wages: np.ndarray,   # (N,) endogenous wages
    q_H: np.ndarray,     # (N, N) per-H-firm qty
    q_L: np.ndarray,     # (N, N) per-L-firm qty
    p: Params,
) -> tuple[float, float, float, float]:
    """
    2×2 payoff matrix (a_SS, a_SL, a_LS, a_LL).

    a_SS = W_S^base + t · Q̄^prod_{H,S} / P̄_S
    a_SL = a_SS    + τ · q̄^imp_{S←L}  / P̄_S + τ_BA · q̄^{H,imp}_{S←L} / P̄_S
    a_LS = a_LL    = W_L^base

    Fiscal cost of being lax-next-to-strict is transmitted through lower lax wages
    (indirect channel) rather than an explicit subtraction.
    """
    strict_mask = sigma == 1
    lax_mask    = sigma == 0

    def _mean(arr, mask):
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    # ── Aggregate state variables ────────────────────────────────────────────
    P_safe = np.maximum(P, 1.0)

    w_S  = _mean(wages,   strict_mask)
    w_L  = _mean(wages,   lax_mask)
    p_S  = _mean(p_star,  strict_mask)
    p_L  = _mean(p_star,  lax_mask)
    fH_S = _mean(f_H,     strict_mask)
    fH_L = _mean(f_H,     lax_mask)
    P_S  = _mean(P,       strict_mask)
    P_L  = _mean(P,       lax_mask)

    inv_PS = 1.0 / max(P_S, 1e-9)
    inv_PL = 1.0 / max(P_L, 1e-9)

    # ── Base welfare terms ───────────────────────────────────────────────────
    cs_S  = (p.a - p_S) ** 2 / (2 * p.b)
    cs_L  = (p.a - p_L) ** 2 / (2 * p.b)

    W_S_base = w_S + cs_S - p.delta * fH_S * inv_PS
    W_L_base = w_L + cs_L - p.delta * fH_L * inv_PL

    # ── Carbon-tax revenue averaged over strict jurisdictions (eq. 3.29) ────
    # Q̄^prod_{H,S} = mean over strict i of [f_H[i] · total_prod_per_H_firm_i]
    N        = len(f_H)
    prod_vals = []
    for i in range(N):
        if sigma[i] != 1:
            continue
        prod_H_i = float(q_H[i, i])
        for m in range(N):
            if m != i and W[m, i] > 0:
                prod_H_i += float(q_H[m, i]) / float(W[m, i])
        prod_vals.append(f_H[i] * prod_H_i)
    Q_prod_HS = float(np.mean(prod_vals)) if prod_vals else 0.0

    # ── Import quantities from lax neighbors into strict jurisdictions ───────
    # Iterate over all (strict i, lax ℓ) neighbor pairs
    imp_all   = []   # total imports per (i, ℓ) pair
    imp_H_all = []   # H-firm imports per (i, ℓ) pair
    for i in range(N):
        if sigma[i] != 1:
            continue
        for ell in range(N):
            if W[i, ell] == 0 or ell == i:
                continue
            if sigma[ell] != 0:
                continue
            imp   = f_H[ell] * float(q_H[i, ell]) + f_L[ell] * float(q_L[i, ell])
            imp_H = f_H[ell] * float(q_H[i, ell])
            imp_all.append(imp)
            imp_H_all.append(imp_H)
    q_imp_SL   = float(np.mean(imp_all))   if imp_all   else 0.0
    q_H_imp_SL = float(np.mean(imp_H_all)) if imp_H_all else 0.0

    # ── Payoff entries ───────────────────────────────────────────────────────
    a_SS = W_S_base + p.t * Q_prod_HS * inv_PS
    a_SL = a_SS     + p.tau * q_imp_SL * inv_PS + p.tau_BA * q_H_imp_SL * inv_PS
    a_LS = W_L_base          # no explicit tariff penalty — transmitted via wages
    a_LL = W_L_base

    return a_SS, a_SL, a_LS, a_LL


# ---------------------------------------------------------------------------
# Network correction — eq. (3.33)
# ---------------------------------------------------------------------------

def network_correction(
    a_SS: float, a_SL: float, a_LS: float, a_LL: float,
    k: float,
) -> float:
    """b_SL entry of the Ohtsuki (2006) correction matrix (eq. 3.33)."""
    if k <= 2:
        return 0.0
    return float(((k + 3) * a_SS + 3 * a_SL - 3 * a_LS - (k + 3) * a_LL)
                 / ((k + 3) * (k - 2)))


# ---------------------------------------------------------------------------
# Policy replicator — eq. (3.34)
# ---------------------------------------------------------------------------

def policy_replicator(
    s: float,
    a_SS: float, a_SL: float, a_LS: float, a_LL: float,
    b_SL: float,
    dt: float,
) -> float:
    """Euler step for s (eq. 3.34).  ṡ = s(1−s)(Π̃_S − Π̃_L)."""
    a_tilde_SS = a_SS
    a_tilde_SL = a_SL + b_SL
    a_tilde_LS = a_LS - b_SL
    a_tilde_LL = a_LL

    Pi_S  = s * a_tilde_SS + (1 - s) * a_tilde_SL
    Pi_L  = s * a_tilde_LS + (1 - s) * a_tilde_LL
    s_dot = s * (1 - s) * (Pi_S - Pi_L)
    return float(np.clip(s + dt * s_dot, 0.0, 1.0))
