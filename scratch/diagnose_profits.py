"""
diagnose_profits.py
-------------------
Run one simulation and print pi_H_bar, pi_L_bar, and h(t) every 50 periods.
Also checks the static profit gap at h=0.5, h=0.6, h=0.7, h=1.0 to find
where the emission replicator's fixed point is.
"""

import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, payoff_matrix, network_correction, policy_replicator,
)

T_SIM  = 500
BURNIN = 200

p = Params(
    N=50, k=3, topology="er",
    M=500, c_H=4.0, c_L=6.0, F=0.0,
    a=20.0, b=1.0,
    t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
    delta_loc=2000.0, delta_glob=500.0,
    mu=0.10, lam=2.0, kappa=1.0,
    relocate=True,
    dt=1.0, T=T_SIM, seed=42,
    mu_P=7.0, sigma_P=1.5,
    s0=0.5, h0=0.5,
)

rng       = np.random.default_rng(p.seed)
P_pop     = np.exp(rng.normal(p.mu_P, p.sigma_P, size=p.N))
sigma     = (rng.random(p.N) < p.s0).astype(int)
firm_loc  = rng.integers(0, p.N, size=p.M)
firm_type = (rng.random(p.M) < p.h0).astype(int)
G, W      = build_network(p.N, p.k, p.topology, p.seed)
k_eff     = effective_degree(G)

print(f"{'t':>5}  {'h':>7}  {'s':>7}  {'piH_bar':>10}  {'piL_bar':>10}  {'piH-piL':>10}  {'h_dot':>10}")
print("-" * 70)

for period in range(1, T_SIM + 1):
    f_H, f_L            = count_firms(firm_loc, firm_type, p.N)
    p_star, q_H, q_L    = solve_market(f_H, f_L, sigma, P_pop, W, p)
    TR                  = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
    pi_H, pi_L          = firm_variable_profits(q_H, q_L, P_pop, p)
    pi_H_bar, pi_L_bar  = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)
    firm_loc            = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W)

    h_old  = float(np.mean(firm_type))
    h_dot  = h_old * (1 - h_old) * (pi_H_bar - pi_L_bar)
    h_new  = emission_replicator(h_old, pi_H_bar, pi_L_bar, p.dt)
    gap    = h_new - h_old
    if abs(gap) > 0.5 / p.M:
        if gap > 0:
            idx = np.where(firm_type == 0)[0]
            n   = min(int(round(gap * p.M)), len(idx))
            if n > 0: firm_type[rng.choice(idx, n, replace=False)] = 1
        else:
            idx = np.where(firm_type == 1)[0]
            n   = min(int(round(-gap * p.M)), len(idx))
            if n > 0: firm_type[rng.choice(idx, n, replace=False)] = 0

    a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P_pop, p_star, W, q_H, q_L, p)
    b_SL  = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
    s_old = float(np.mean(sigma))
    s_new = policy_replicator(s_old, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt, p.kappa)
    gap_s    = s_new - s_old
    n_change = int(round(abs(gap_s) * p.N))
    if n_change > 0:
        if gap_s > 0:
            cands = np.where(sigma == 0)[0]
            n = min(n_change, len(cands))
            if n > 0: sigma[rng.choice(cands, n, replace=False)] = 1
        else:
            cands = np.where(sigma == 1)[0]
            n = min(n_change, len(cands))
            if n > 0: sigma[rng.choice(cands, n, replace=False)] = 0

    if period % 25 == 0 or period <= 5:
        h_act = float(np.mean(firm_type))
        s_act = float(np.mean(sigma))
        print(f"{period:>5}  {h_act:>7.4f}  {s_act:>7.4f}  "
              f"{pi_H_bar:>10.4f}  {pi_L_bar:>10.4f}  "
              f"{pi_H_bar - pi_L_bar:>10.4f}  {h_dot:>10.6f}")

# ── Static sweep: what is pi_H_bar - pi_L_bar as a function of h? ─────────
print()
print("Static profit gap  Δπ = π_H_bar − π_L_bar  vs h  (sigma fixed at s=0.5)")
print(f"{'h_init':>7}  {'piH_bar':>10}  {'piL_bar':>10}  {'delta_pi':>10}  {'h_dot_sign':>12}")
print("-" * 58)

rng2 = np.random.default_rng(99)
P2   = np.exp(rng2.normal(p.mu_P, p.sigma_P, size=p.N))
sig2 = (rng2.random(p.N) < 0.5).astype(int)
G2, W2 = build_network(p.N, p.k, p.topology, seed=99)

for h_test in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    rng3      = np.random.default_rng(77)
    fl        = rng3.integers(0, p.N, size=p.M)
    ft        = (rng3.random(p.M) < h_test).astype(int)
    fH, fL    = count_firms(fl, ft, p.N)
    _, qH, qL = solve_market(fH, fL, sig2, P2, W2, p)
    piH, piL  = firm_variable_profits(qH, qL, P2, p)
    piHb, piLb = average_profits(fl, ft, piH, piL, p.F)
    dpi  = piHb - piLb
    sign = "→ h↑" if dpi > 0 else ("→ h↓" if dpi < 0 else "= fixed pt")
    print(f"{h_test:>7.1f}  {piHb:>10.4f}  {piLb:>10.4f}  {dpi:>10.4f}  {sign:>12}")
