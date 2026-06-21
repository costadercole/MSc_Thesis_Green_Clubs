"""
trajectory.py — single coevolutionary trajectory, §3.9.

Steps each period:
  1. Solve goods market
  2. Compute fiscal revenues and per-capita welfare
  3. Relocate H-firms (eq. 3.36, rate mu)
  4. Update firm emission types via Fermi imitation (eq. 3.42, rate nu)
  5. Update jurisdiction policies via Fermi imitation (eq. 3.41, rate lam)
"""

import numpy as np
from params import Params
from model.network import build_network
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, firm_type_update
from model.jurisdictions import (
    fiscal_revenues, per_capita_welfare, fermi_policy_update,
)


def run_trajectory(
    mu, lam, delta_loc, tau_BA,
    h0, s0,
    T=1000, seed=42,
    c_H=4.0, c_L=5.5, t=4.0, tau=0.5, g=2.2,
    delta_glob=2000.0, kappa=1.0,
    nu=2.0, kappa_f=1e-3, eps=1e-2,
    N=50, k=3, topology="er", M=500,
    mu_P=7.0, sigma_P=1.5,
    relocate=True,
    return_series=False,
):
    p = Params(
        N=N, k=k, topology=topology, M=M,
        c_H=c_H, c_L=c_L, F=0.0, a=20.0, b=1.0,
        t=t, tau=tau, g=g, tau_BA=tau_BA,
        delta_loc=delta_loc, delta_glob=delta_glob,
        mu=mu, lam=lam, nu=nu, kappa=kappa, kappa_f=kappa_f, eps=eps,
        relocate=relocate,
        dt=0.05, T=T, seed=seed,
        mu_P=mu_P, sigma_P=sigma_P,
        s0=s0, h0=h0,
    )

    rng       = np.random.default_rng(p.seed)
    P_pop     = np.exp(rng.normal(p.mu_P, p.sigma_P, size=p.N))
    sigma     = (rng.random(p.N) < p.s0).astype(int)
    firm_loc  = rng.integers(0, p.N, size=p.M)
    firm_type = (rng.random(p.M) < p.h0).astype(int)
    G, W      = build_network(p.N, p.k, p.topology, p.seed)

    h_ser = np.empty(T)
    s_ser = np.empty(T)

    for period in range(T):
        f_H, f_L         = count_firms(firm_loc, firm_type, p.N)
        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P_pop, W, p)
        pi_H, pi_L       = firm_variable_profits(q_H, q_L, P_pop, p)
        TR               = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        welfare          = per_capita_welfare(f_H, sigma, P_pop, p_star, TR, p, q_H)

        profit = np.where(firm_type == 1, pi_H[firm_loc], pi_L[firm_loc])

        firm_loc  = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W=W)
        firm_type = firm_type_update(firm_type, profit, nu=p.nu, kappa_f=p.kappa_f,
                                     dt=p.dt, rng=rng, eps=p.eps)

        f_H, f_L         = count_firms(firm_loc, firm_type, p.N)
        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P_pop, W, p)
        TR               = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        welfare          = per_capita_welfare(f_H, sigma, P_pop, p_star, TR, p, q_H)
        sigma            = fermi_policy_update(sigma, welfare, W, p, rng)

        h_ser[period] = float(np.mean(firm_type))
        s_ser[period] = float(np.mean(sigma))

    half    = T // 2
    h_ss    = float(np.mean(h_ser[half:]))
    s_ss    = float(np.mean(s_ser[half:]))
    h_final = float(h_ser[-1])
    s_final = float(s_ser[-1])

    if return_series:
        return h_ss, s_ss, h_final, s_final, h_ser, s_ser
    return h_ss, s_ss, h_final, s_final
