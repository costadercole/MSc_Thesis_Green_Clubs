"""
trajectory.py
-------------
Single source of truth for running one coevolutionary trajectory and
returning its endpoint. Used by all diagnostic scripts in this batch
(gc_stability.py, basin_movement.py, scale_invariance.py) so the dynamics
are never reimplemented or altered between diagnostics.

This mirrors the simulation loop in run_model.py / find_regimes.py exactly:
no changes to the relocation rule, emission replicator, payoff matrix, or
policy replicator.
"""

import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, payoff_matrix, network_correction, policy_replicator,
)


def run_trajectory(
    mu, lam, delta_loc, tau_BA,
    h0, s0,
    T=1000, seed=42,
    c_H=4.0, c_L=6.0, t=5.0, tau=3.0, g=2.4,
    delta_glob=500.0, kappa=1.0,
    N=50, k=3, topology="er", M=500,
    mu_P=7.0, sigma_P=1.5,
    relocate=True,
    return_series=False,
):
    """
    Run one full coevolutionary trajectory and return the endpoint.

    Returns
    -------
    h_ss, s_ss : float
        Mean of h(t), s(t) over the second half of the run (steady-state proxy).
    h_final, s_final : float
        Value at the very last period.
    (h_ser, s_ser) : np.ndarray, np.ndarray   [only if return_series=True]
    """
    p = Params(
        N=N, k=k, topology=topology, M=M,
        c_H=c_H, c_L=c_L, F=0.0, a=20.0, b=1.0,
        t=t, tau=tau, g=g, tau_BA=tau_BA,
        delta_loc=delta_loc, delta_glob=delta_glob,
        mu=mu, lam=lam, kappa=kappa,
        relocate=relocate,
        dt=1.0, T=T, seed=seed,
        mu_P=mu_P, sigma_P=sigma_P,
        s0=s0, h0=h0,
    )

    rng       = np.random.default_rng(p.seed)
    P_pop     = np.exp(rng.normal(p.mu_P, p.sigma_P, size=p.N))
    sigma     = (rng.random(p.N) < p.s0).astype(int)
    firm_loc  = rng.integers(0, p.N, size=p.M)
    firm_type = (rng.random(p.M) < p.h0).astype(int)
    G, W      = build_network(p.N, p.k, p.topology, p.seed)
    k_eff     = effective_degree(G)

    h_ser = np.empty(T)
    s_ser = np.empty(T)

    for period in range(1, T + 1):
        f_H, f_L            = count_firms(firm_loc, firm_type, p.N)
        p_star, q_H, q_L    = solve_market(f_H, f_L, sigma, P_pop, W, p)
        TR                  = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        pi_H, pi_L          = firm_variable_profits(q_H, q_L, P_pop, p)
        pi_H_bar, pi_L_bar  = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)
        firm_loc            = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W)

        h_old = float(np.mean(firm_type))
        h_new = emission_replicator(h_old, pi_H_bar, pi_L_bar, p.dt)
        gap   = h_new - h_old
        if abs(gap) > 0.5 / p.M:
            if gap > 0:
                idx = np.where(firm_type == 0)[0]
                n   = min(int(round(gap * p.M)), len(idx))
                if n > 0:
                    firm_type[rng.choice(idx, n, replace=False)] = 1
            else:
                idx = np.where(firm_type == 1)[0]
                n   = min(int(round(-gap * p.M)), len(idx))
                if n > 0:
                    firm_type[rng.choice(idx, n, replace=False)] = 0

        a_SS, a_SL, a_LS, a_LL = payoff_matrix(
            f_H, f_L, sigma, P_pop, p_star, W, q_H, q_L, p)
        b_SL  = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
        s_old = float(np.mean(sigma))
        s_new = policy_replicator(s_old, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt, p.kappa, p.lam)
        gap_s    = s_new - s_old
        n_change = int(round(abs(gap_s) * p.N))
        if n_change > 0:
            if gap_s > 0:
                cands = np.where(sigma == 0)[0]
                n = min(n_change, len(cands))
                if n > 0:
                    sigma[rng.choice(cands, n, replace=False)] = 1
            else:
                cands = np.where(sigma == 1)[0]
                n = min(n_change, len(cands))
                if n > 0:
                    sigma[rng.choice(cands, n, replace=False)] = 0

        h_ser[period - 1] = float(np.mean(firm_type))
        s_ser[period - 1] = float(np.mean(sigma))

    half = T // 2
    h_ss    = float(np.mean(h_ser[half:]))
    s_ss    = float(np.mean(s_ser[half:]))
    h_final = float(h_ser[-1])
    s_final = float(s_ser[-1])

    if return_series:
        return h_ss, s_ss, h_final, s_final, h_ser, s_ser
    return h_ss, s_ss, h_final, s_final
