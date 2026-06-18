"""
Jurisdiction state and dynamics.

State
-----
sigma : (N,) int  — regulatory policy: 1 = strict (S), 0 = lax (L)

Fiscal revenues (eqs. 3.22–3.24)
---------------------------------
TR_i^tax     = t·1[S]·f_H[i]·Σ_m q_H[m,i]          (carbon tax on total H-output in i)
TR_i^tariff  = τ·1[S]·Σ_{ℓ lax nbr} (f_H[ℓ]·q_H[i,ℓ] + f_L[ℓ]·q_L[i,ℓ])
TR_i^BCA     = τ_BA·1[S]·Σ_{ℓ lax nbr} f_H[ℓ]·q_H[i,ℓ]

Per-capita welfare (eq. 3.40):
W_i/P_i = (a−p*_i)²/(2b) + TR_i/P_i − D_i
where D_i = δ_loc·f_H[i]/P_i + δ_glob·(Σ_j f_H[j])/(Σ_j P_j)

Payoff matrix (eqs. 3.43–3.47):
a_SS = W_S^base + t·Q̄^prod_{H,S}/P̄_S
a_SL = a_SS + τ·q̄^imp_{S←L}/P̄_S + τ_BA·q̄^{H,imp}_{S←L}/P̄_S
a_LS = W_L^base − Φ^exp_{L→S}      (export-side penalty, eq. 3.47)
a_LL = W_L^base
where W_X^base = cs_X − δ_loc·f̄^H_X/P̄_X − δ_glob·f̄^H_total/P̄_total

Policy dynamics (eqs. 3.51, 3.53):
b_SL = (a_SS + a_SL − a_LS − a_LL) / (k−2)   [pairwise comparison]
ṡ = s(1−s)·tanh[κ/2·(Π̃_S − Π̃_L)]           [Fermi replicator]
"""

import copy
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
    W: np.ndarray,      # (N, N) neighbour weight matrix
    q_H: np.ndarray,    # (N, N) per-H-firm delivered qty from j to i
    q_L: np.ndarray,    # (N, N) per-L-firm delivered qty from j to i
    p: Params,
) -> np.ndarray:
    """
    Total fiscal revenue TR_i for each jurisdiction (eq. 3.24).

    With additive transport costs (no iceberg melt), quantity produced at
    origin j for market i equals q_H[i, j] exactly, so no iceberg correction.
    """
    N  = len(f_H)
    TR = np.zeros(N)

    for i in range(N):
        if sigma[i] == 0:
            continue   # lax: no direct fiscal revenue

        # Carbon-tax revenue (eq. 3.22): t · f_H[i] · total H-output from i
        total_H_output_i = sum(float(q_H[m, i]) for m in range(N))
        TR[i] += p.t * f_H[i] * total_H_output_i

        # Tariff + BCA revenue (eqs. 3.23–3.24): charged on lax-neighbour imports
        for ell in range(N):
            if W[i, ell] == 0 or ell == i:
                continue
            if sigma[ell] != 0:
                continue   # tariff only on imports from lax origins
            q_tot_imp = f_H[ell] * float(q_H[i, ell]) + f_L[ell] * float(q_L[i, ell])
            q_H_imp   = f_H[ell] * float(q_H[i, ell])
            TR[i] += p.tau    * q_tot_imp
            TR[i] += p.tau_BA * q_H_imp

    return TR


# ---------------------------------------------------------------------------
# Per-capita welfare — eq. (3.40)
# ---------------------------------------------------------------------------

def per_capita_welfare(
    f_H: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    p_star: np.ndarray,
    TR: np.ndarray,
    p: Params,
) -> np.ndarray:
    """
    Per-capita welfare W_i/P_i (eq. 3.40).

    = (a−p*_i)²/(2b)  +  TR_i/P_i  −  D_i
    D_i = δ_loc·f_H[i]/P_i + δ_glob·(Σ f_H)/(Σ P)
    """
    P_safe      = np.maximum(P, 1.0)
    cs          = (p.a - p_star) ** 2 / (2 * p.b)
    fiscal      = TR / P_safe
    local_dmg   = p.delta_loc  * f_H / P_safe
    global_dmg  = p.delta_glob * f_H.sum() / P_safe.sum()   # scalar, same for all i
    return cs + fiscal - local_dmg - global_dmg


# ---------------------------------------------------------------------------
# Export-side penalty — eq. (3.47)
# ---------------------------------------------------------------------------

def _export_side_penalty(
    f_H: np.ndarray,
    f_L: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    W: np.ndarray,
    pi_H_current: np.ndarray,   # (N,) H-firm variable profit under current tariffs
    p: Params,
) -> float:
    """
    Φ^exp_{L→S}: per-capita profit loss for H-firms in lax jurisdictions
    due to tariffs imposed by strict neighbours (eq. 3.47).

    Computed as mean over lax jurisdictions of:
        f_H[j] * (pi_H_notariff[j] - pi_H_current[j]) / P[j]

    where pi_H_notariff is H-firm profits with τ=0 and τ_BA=0.
    """
    from model.market import solve_market, firm_variable_profits

    lax_mask = sigma == 0
    if not lax_mask.any():
        return 0.0

    # Counterfactual: remove tariff and BCA
    p0 = copy.copy(p)
    p0.tau    = 0.0
    p0.tau_BA = 0.0
    _, q_H_0, q_L_0 = solve_market(f_H, f_L, sigma, P, W, p0)
    pi_H_0, _ = firm_variable_profits(q_H_0, q_L_0, P, p0)

    P_safe = np.maximum(P, 1.0)
    phi_vals = [
        f_H[j] * float(pi_H_0[j] - pi_H_current[j]) / float(P_safe[j])
        for j in range(len(sigma)) if sigma[j] == 0
    ]
    return float(np.mean(phi_vals)) if phi_vals else 0.0


# ---------------------------------------------------------------------------
# Payoff matrix — eqs. (3.43)–(3.47)
# ---------------------------------------------------------------------------

def payoff_matrix(
    f_H: np.ndarray,
    f_L: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    p_star: np.ndarray,
    W: np.ndarray,
    q_H: np.ndarray,
    q_L: np.ndarray,
    p: Params,
) -> tuple[float, float, float, float]:
    """
    2×2 payoff matrix (a_SS, a_SL, a_LS, a_LL).

    a_SS = W_S^base + t · Q̄^prod_{H,S} / P̄_S
    a_SL = a_SS    + τ · q̄^imp_{S←L}  / P̄_S + τ_BA · q̄^{H,imp}_{S←L} / P̄_S
    a_LS = W_L^base − Φ^exp_{L→S}
    a_LL = W_L^base
    """
    from model.market import firm_variable_profits

    strict_mask = sigma == 1
    lax_mask    = sigma == 0

    def _mean(arr, mask):
        return float(np.mean(arr[mask])) if mask.any() else 0.0

    P_safe = np.maximum(P, 1.0)

    p_S  = _mean(p_star, strict_mask)
    p_L  = _mean(p_star, lax_mask)
    fH_S = _mean(f_H,    strict_mask)
    fH_L = _mean(f_H,    lax_mask)
    P_S  = _mean(P,      strict_mask)
    P_L  = _mean(P,      lax_mask)

    inv_PS = 1.0 / max(P_S, 1e-9)
    inv_PL = 1.0 / max(P_L, 1e-9)

    # Global damage component — same for all jurisdictions (cancels in S−L gap,
    # but kept for absolute welfare levels used in the replicator)
    P_safe      = np.maximum(P, 1.0)
    glob_damage = p.delta_glob * f_H.sum() / P_safe.sum()

    # Base welfare terms (eq. 3.43–3.44): CS − local damage − global damage
    cs_S  = (p.a - p_S) ** 2 / (2 * p.b)
    cs_L  = (p.a - p_L) ** 2 / (2 * p.b)

    W_S_base = cs_S - p.delta_loc * fH_S * inv_PS - glob_damage
    W_L_base = cs_L - p.delta_loc * fH_L * inv_PL - glob_damage

    # Carbon-tax revenue averaged over strict jurisdictions (eq. 3.45)
    N = len(f_H)
    prod_vals = []
    for i in range(N):
        if sigma[i] != 1:
            continue
        total_H_output_i = sum(float(q_H[m, i]) for m in range(N))
        prod_vals.append(f_H[i] * total_H_output_i)
    Q_prod_HS = float(np.mean(prod_vals)) if prod_vals else 0.0

    # Import quantities from lax neighbors into strict jurisdictions (eq. 3.46)
    imp_all   = []
    imp_H_all = []
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

    # Export-side penalty for lax jurisdictions (eq. 3.47)
    pi_H_curr, _ = firm_variable_profits(q_H, q_L, P, p)
    phi_exp = _export_side_penalty(f_H, f_L, sigma, P, W, pi_H_curr, p)

    # Payoff entries
    a_SS = W_S_base + p.t * Q_prod_HS * inv_PS
    a_SL = a_SS     + p.tau * q_imp_SL * inv_PS + p.tau_BA * q_H_imp_SL * inv_PS
    a_LS = W_L_base - phi_exp
    a_LL = W_L_base

    return a_SS, a_SL, a_LS, a_LL


# ---------------------------------------------------------------------------
# Network correction — eq. (3.51)  pairwise comparison
# ---------------------------------------------------------------------------

def network_correction(
    a_SS: float, a_SL: float, a_LS: float, a_LL: float,
    k: float,
) -> float:
    """b_SL for pairwise-comparison imitation (eq. 3.51)."""
    if k <= 2:
        return 0.0
    return float((a_SS + a_SL - a_LS - a_LL) / (k - 2))


# ---------------------------------------------------------------------------
# Policy replicator — eq. (3.53)  Fermi rule
# ---------------------------------------------------------------------------

def policy_replicator(
    s: float,
    a_SS: float, a_SL: float, a_LS: float, a_LL: float,
    b_SL: float,
    dt: float,
    kappa: float,
    lam: float = 1.0,
) -> float:
    """
    Euler step for s (eq. 3.53), scaled by institutional adaptation rate λ.

    ṡ = λ · s(1−s)·tanh[κ/2·(Π̃_S − Π̃_L)]
    """
    a_tilde_SS = a_SS
    a_tilde_SL = a_SL + b_SL
    a_tilde_LS = a_LS - b_SL
    a_tilde_LL = a_LL

    Pi_S  = s * a_tilde_SS + (1 - s) * a_tilde_SL
    Pi_L  = s * a_tilde_LS + (1 - s) * a_tilde_LL
    s_dot = s * (1 - s) * float(np.tanh(kappa / 2.0 * (Pi_S - Pi_L)))
    return float(np.clip(s + lam * dt * s_dot, 0.0, 1.0))
