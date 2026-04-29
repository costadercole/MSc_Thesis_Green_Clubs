"""
Main simulation loop — §3.9.

At each time step:
  (i)   compute equilibrium prices p*_i  (market.py)
  (ii)  compute firm profits and jurisdiction welfare
  (iii) update firm locations (relocation)
  (iv)  update h and s via replicator equations

Returns a dict of time-series arrays for downstream analysis.
"""

import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.firms import (
    init_firms, count_firms,
    relocate_firms, emission_replicator, average_profits,
)
from model.jurisdictions import (
    init_jurisdictions, init_populations,
    payoff_matrix, network_correction, policy_replicator,
)
from model.market import (
    equilibrium_prices, firm_variable_profits,
)


def run(p: Params) -> dict:
    """
    Run the full simulation and return recorded time series.

    Keys in the returned dict
    -------------------------
    t       : (T,) time points
    h       : (T,) share of high-emission firms
    s       : (T,) share of strict jurisdictions
    f_H     : (T, N) high-emission firm counts per jurisdiction
    f_L     : (T, N) low-emission firm counts per jurisdiction
    p_star  : (T, N) equilibrium prices
    sigma   : (T, N) regulatory policy (int)
    P       : (N,)  jurisdiction populations (fixed)
    W       : (N, N) trade weight matrix
    """
    rng = np.random.default_rng(p.seed)

    # --- Build network ---
    G, W = build_network(p.N, p.k, p.topology, p.seed)
    k_eff = effective_degree(G)

    # --- Initialise state ---
    P          = init_populations(p, rng)
    sigma      = init_jurisdictions(p, rng)
    firm_loc, firm_type = init_firms(p, rng)

    h = float(np.mean(firm_type))
    s = float(np.mean(sigma))

    # --- Pre-allocate storage ---
    T = p.T
    rec_h      = np.empty(T)
    rec_s      = np.empty(T)
    rec_f_H    = np.empty((T, p.N))
    rec_f_L    = np.empty((T, p.N))
    rec_p_star = np.empty((T, p.N))
    rec_sigma  = np.empty((T, p.N), dtype=int)

    # --- Main loop ---
    for step in range(T):
        # (i) equilibrium prices
        f_H, f_L = count_firms(firm_loc, firm_type, p.N)
        p_star   = equilibrium_prices(f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.a, p.b)

        # (ii) firm profits
        pi_H, pi_L = firm_variable_profits(p_star, f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.b)

        # (iii) relocate firms, then recompute prices and profits
        firm_loc   = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng)
        f_H, f_L   = count_firms(firm_loc, firm_type, p.N)
        p_star     = equilibrium_prices(f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.a, p.b)
        pi_H, pi_L = firm_variable_profits(p_star, f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.b)

        # (iv-a) emission replicator → update firm types
        pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)
        h = emission_replicator(h, pi_H_bar, pi_L_bar, p.dt)
        _apply_h(firm_type, h, rng)

        # (iv-b) policy replicator → update sigma
        a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P, p_star, W, p)
        b_SL = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
        s = policy_replicator(s, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt)
        _apply_s(sigma, s, rng)

        # Record
        rec_h[step]      = h
        rec_s[step]      = s
        rec_f_H[step]    = f_H
        rec_f_L[step]    = f_L
        rec_p_star[step] = p_star
        rec_sigma[step]  = sigma.copy()

    return {
        "t":      np.arange(T) * p.dt,
        "h":      rec_h,
        "s":      rec_s,
        "f_H":    rec_f_H,
        "f_L":    rec_f_L,
        "p_star": rec_p_star,
        "sigma":  rec_sigma,
        "P":      P,
        "W":      W,
    }


# ---------------------------------------------------------------------------
# Helpers: reconcile continuous replicator values with discrete agent arrays
# ---------------------------------------------------------------------------

def _apply_h(firm_type: np.ndarray, h_target: float, rng: np.random.Generator) -> None:
    """Convert firm types so fraction of H-firms matches h_target (in-place)."""
    M = len(firm_type)
    gap = h_target - float(np.mean(firm_type))
    if gap > 0:
        idx = np.where(firm_type == 0)[0]
        n = min(int(round(gap * M)), len(idx))
        if n > 0:
            firm_type[rng.choice(idx, n, replace=False)] = 1
    elif gap < 0:
        idx = np.where(firm_type == 1)[0]
        n = min(int(round(-gap * M)), len(idx))
        if n > 0:
            firm_type[rng.choice(idx, n, replace=False)] = 0


def _apply_s(sigma: np.ndarray, s_target: float, rng: np.random.Generator) -> None:
    """Flip jurisdiction policies so fraction of strict matches s_target (in-place)."""
    N = len(sigma)
    gap = s_target - float(np.mean(sigma))
    if gap > 0:
        idx = np.where(sigma == 0)[0]
        n = min(int(round(gap * N)), len(idx))
        if n > 0:
            sigma[rng.choice(idx, n, replace=False)] = 1
    elif gap < 0:
        idx = np.where(sigma == 1)[0]
        n = min(int(round(-gap * N)), len(idx))
        if n > 0:
            sigma[rng.choice(idx, n, replace=False)] = 0
