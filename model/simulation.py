"""
Main simulation loop — §3.9.

At each time step:
  1. Solve Cournot market given current firm_type, firm_loc, sigma
  2. Compute per-firm profits and jurisdiction welfare (incl. output-based damage)
  3. Relocate H-firms to neighbouring jurisdictions (eq. 3.36, rate mu)
  4. Update firm emission types via agent-level Fermi imitation (eq. 3.42, rate nu)
  5. Update jurisdiction policies via agent-level Fermi imitation (eq. 3.41, rate lam)

All three stochastic processes (mu, nu, lam) operate on the same dt-scaled
Poisson rate, so their relative speeds are directly controlled by mu/lam and nu/lam.
"""

import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.firms import init_firms, count_firms, relocate_firms, firm_type_update
from model.jurisdictions import (
    init_jurisdictions, init_populations,
    fiscal_revenues, per_capita_welfare,
    fermi_policy_update,
)
from model.market import solve_market, firm_variable_profits


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

    G, W = build_network(p.N, p.k, p.topology, p.seed)

    P                   = init_populations(p, rng)
    sigma               = init_jurisdictions(p, rng)
    firm_loc, firm_type = init_firms(p, rng)

    T = p.T
    rec_h      = np.empty(T)
    rec_s      = np.empty(T)
    rec_f_H    = np.empty((T, p.N))
    rec_f_L    = np.empty((T, p.N))
    rec_p_star = np.empty((T, p.N))
    rec_sigma  = np.empty((T, p.N), dtype=int)

    # Burn-in: relocation only, no type-switching or policy updates.
    # Lets H-firms sort into lax jurs before the coupled dynamics begin.
    T_burnin = getattr(p, "T_burnin", 0)
    for _ in range(T_burnin):
        f_H, f_L = count_firms(firm_loc, firm_type, p.N)
        _, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, p)
        pi_H, _ = firm_variable_profits(q_H, q_L, P, p)
        firm_loc = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W=W)

    for step in range(T):
        # Step 1: solve goods market
        f_H, f_L = count_firms(firm_loc, firm_type, p.N)
        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, p)

        # Step 2: per-firm profits and jurisdiction welfare
        pi_H, pi_L = firm_variable_profits(q_H, q_L, P, p)
        TR         = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        welfare    = per_capita_welfare(f_H, sigma, P, p_star, TR, p, q_H, f_L, q_L)

        # Build per-firm profit vector: each firm earns the profit of its type
        # in its current jurisdiction (pi_H or pi_L indexed by firm_loc).
        profit = np.where(
            firm_type == 1,
            pi_H[firm_loc],   # H-firm profit in its jurisdiction
            pi_L[firm_loc],   # L-firm profit in its jurisdiction
        )

        # Step 3: relocate H-firms (eq. 3.36, rate mu)
        firm_loc = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W=W)

        # Step 4: firm type update — Fermi imitation from global pool (eq. 3.42, rate nu)
        firm_type = firm_type_update(
            firm_type, profit,
            nu=p.nu, kappa_f=p.kappa_f, dt=p.dt, rng=rng, eps=p.eps,
        )
        h = float(np.mean(firm_type))

        # Step 5: jurisdiction policy update — Fermi imitation (eq. 3.41, rate lam)
        # Recompute market after relocation + type changes for accurate welfare
        f_H, f_L = count_firms(firm_loc, firm_type, p.N)
        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, p)
        TR      = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        welfare = per_capita_welfare(f_H, sigma, P, p_star, TR, p, q_H, f_L, q_L)
        sigma   = fermi_policy_update(sigma, welfare, W, p, rng)
        s       = float(np.mean(sigma))

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
