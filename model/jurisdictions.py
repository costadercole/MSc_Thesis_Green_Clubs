"""
Jurisdiction state and dynamics — §3.3, §3.7, §3.8.

State
-----
sigma : (N,) int  — regulatory policy: 1 = strict (S), 0 = lax (L)

Fiscal revenues (eqs. 3.29–3.32)
---------------------------------
TR_i^tax     = t · 1[S] · Σ_{m} f_H[i] · q_H[m, i]
TR_i^tariff  = τ · 1[S] · Σ_{ℓ lax nbr} (f_H[ℓ]·q_H[i,ℓ] + f_L[ℓ]·q_L[i,ℓ])
TR_i^BCA     = τ_BA · 1[S] · Σ_{ℓ lax nbr} f_H[ℓ]·q_H[i,ℓ]

Per-capita welfare (eq. 3.40):
W_i/P_i = (a−p*_i)²/(2b) + TR_i/P_i − D_i
where D_i = δ_loc·f_H[i]/P_i + δ_glob·(Σ_j f_H[j])/(Σ_j P_j)

Policy dynamics (eq. 3.41):
Each jurisdiction i samples one neighbour j ~ N(i) with prob λ·Δt and
switches σ_i ← σ_j with Fermi probability 1 / (1 + exp(-κ(W_j/P_j − W_i/P_i))).
"""

import numpy as np
from scipy.special import expit
from params import Params


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_jurisdictions(p: Params, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(p.N) < p.s0).astype(int)


def init_populations(p: Params, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.normal(p.mu_P, p.sigma_P, size=p.N))


# ---------------------------------------------------------------------------
# Fiscal revenues — eqs. (3.29)–(3.32)
# ---------------------------------------------------------------------------

def fiscal_revenues(
    f_H: np.ndarray,    # (N,)
    f_L: np.ndarray,    # (N,)
    sigma: np.ndarray,  # (N,)
    W: np.ndarray,      # (N, N) neighbour weight matrix
    q_H: np.ndarray,    # (N, N) per-H-firm delivered qty from j to i  (q_H[i, j])
    q_L: np.ndarray,    # (N, N) per-L-firm delivered qty from j to i  (q_L[i, j])
    p: Params,
) -> np.ndarray:
    """Total fiscal revenue TR_i for each jurisdiction (eq. 3.32)."""
    N  = len(f_H)
    TR = np.zeros(N)

    for i in range(N):
        if sigma[i] == 0:
            continue   # lax: no direct fiscal revenue

        # Carbon-tax revenue (eq. 3.29): t · f_H[i] · total H-output from i
        # Total H output from i = sum over all destination markets m of q_H[m, i]
        total_H_output_i = float(q_H[:, i].sum())
        TR[i] += p.t * f_H[i] * total_H_output_i

        # Tariff + BCA revenue (eqs. 3.30–3.31): charged on lax-neighbour imports
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
    q_H: np.ndarray = None,   # (N, N) per-H-firm delivered qty; if provided, damage ∝ total output
    f_L: np.ndarray = None,   # (N,) L-firm counts; needed for the host-benefit term
    q_L: np.ndarray = None,   # (N, N) per-L-firm delivered qty; needed for the host-benefit term
) -> np.ndarray:
    """
    Per-capita welfare W_i/P_i (eq. 3.40, extended with a host-benefit channel).

    = (a−p*_i)²/(2b)  +  TR_i/P_i  +  B_i  −  D_i
    B_i = φ·(E_H[i] + E_L[i]) / P_i          (host benefit: jobs + tax base)
    D_i = δ_loc·E_H[i]/P_i + δ_glob·(Σ E_H)/(Σ P)

    where E_X[i] = f_X[i] · Σ_m q_X[m,i] is total type-X output produced in i.
    If q_H is not supplied, E_H[i] = f_H[i] (firm-count fallback).

    Host-benefit term (φ ≥ 0)
    -------------------------
    Reduced-form value that a jurisdiction retains from production physically
    located in it — employment/wages, general (non-carbon) tax base, and local
    profit/agglomeration.  Accrues under BOTH policies (it is not an
    environmental instrument), proportional to local output.  This is the
    economic upside that makes hosting industry — including dirty H-firms —
    attractive, so that going strict means losing that activity (and, once
    firms are mobile, losing it to lax neighbours).

    The net per-unit payoff of hosting an H-firm's output is (φ − δ_loc):
      φ < δ_loc → damage dominates → strict preferred (green club)
      φ > δ_loc → jobs/tax dominate → lax preferred (pollution haven)
    φ = 0 recovers the original damage-only welfare.
    """
    P_safe = np.maximum(P, 1.0)
    cs     = (p.a - p_star) ** 2 / (2 * p.b)
    fiscal = TR / P_safe

    if q_H is not None:
        # Total H-firm output per jurisdiction: f_H[i] * (sum of per-firm supply across all markets)
        E_H = f_H * q_H.sum(axis=0)   # q_H[m, i].sum(axis=0) = total per-firm supply from i
    else:
        E_H = f_H

    # Host benefit: jobs + tax base from ALL local production (H and L), under any policy.
    phi = getattr(p, "phi", 0.0)
    if phi != 0.0 and q_L is not None and f_L is not None:
        E_L     = f_L * q_L.sum(axis=0)
        benefit = phi * (E_H + E_L) / P_safe
    else:
        benefit = 0.0

    local_dmg  = p.delta_loc  * E_H / P_safe
    global_dmg = p.delta_glob * E_H.sum() / P_safe.sum()
    return cs + fiscal + benefit - local_dmg - global_dmg


# ---------------------------------------------------------------------------
# Agent-level Fermi policy update — eq. (3.41)
# ---------------------------------------------------------------------------

def fermi_policy_update(
    sigma: np.ndarray,   # (N,) current policies, modified in-place
    welfare: np.ndarray, # (N,) per-capita welfare W_i/P_i
    W: np.ndarray,       # (N, N) network weight matrix (non-zero = neighbour)
    p: Params,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Each jurisdiction i, with probability λ·Δt, samples one neighbour j ~ N(i)
    uniformly at random and switches σ_i ← σ_j with Fermi probability

        P(σ_i ← σ_j) = 1 / (1 + exp(-κ (W_j/P_j − W_i/P_i)))   (eq. 3.41)

    Updates are synchronous: all switching decisions use the welfare vector
    computed before any switches this step.
    """
    sigma_new = sigma.copy()
    N = len(sigma)

    for i in range(N):
        # Each jurisdiction has an independent update event with prob λ·Δt
        if rng.random() >= p.lam * p.dt:
            continue

        # Sample one neighbour j from N(i) (non-zero weight, excluding self)
        neighbours = [j for j in range(N) if W[i, j] > 0 and j != i]
        if not neighbours:
            continue
        j = neighbours[rng.integers(len(neighbours))]

        # Fermi switching probability (expit is stable for large |delta_W|)
        delta_W = welfare[j] - welfare[i]
        prob    = float(expit(p.kappa * delta_W))
        if rng.random() < prob:
            sigma_new[i] = sigma[j]

    return sigma_new
