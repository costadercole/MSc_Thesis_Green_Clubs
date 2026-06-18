"""
diagnose_mx.py
--------------
Run a small set of representative (µ, λ) cells and print the actual
seed-averaged (h_ss, s_ss) values, so we can see what the MX cells
are converging to and decide which parameter knobs to turn.
"""

import sys
import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, per_capita_welfare, payoff_matrix,
    network_correction, policy_replicator,
)

T_SIM  = 1000
BURNIN = 200
SEEDS  = [42, 123, 7, 999, 2024]

def run_sim(mu, lam, delta_loc=2000.0, seed=42):
    p = Params(
        N=50, k=3, topology="er",
        M=500, c_H=4.0, c_L=6.0, F=0.0,
        a=20.0, b=1.0,
        t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
        delta_loc=delta_loc, delta_glob=500.0,
        mu=mu, lam=lam, kappa=1.0,
        relocate=True,
        dt=1.0, T=T_SIM, seed=seed,
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

    h_series = np.empty(T_SIM)
    s_series = np.empty(T_SIM)

    for period in range(1, T_SIM + 1):
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
                if n > 0: firm_type[rng.choice(idx, n, replace=False)] = 1
            else:
                idx = np.where(firm_type == 1)[0]
                n   = min(int(round(-gap * p.M)), len(idx))
                if n > 0: firm_type[rng.choice(idx, n, replace=False)] = 0

        a_SS, a_SL, a_LS, a_LL = payoff_matrix(
            f_H, f_L, sigma, P_pop, p_star, W, q_H, q_L, p)
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

        h_series[period - 1] = float(np.mean(firm_type))
        s_series[period - 1] = float(np.mean(sigma))

    return float(np.mean(h_series[BURNIN:])), float(np.mean(s_series[BURNIN:]))

# ─── Diagnose MX cells: sample from the three MX µ bands ───────────────────
# Plus test δ_loc variation to see which lever moves s toward 1

PROBE_MU_LAM = [
    # MX rows from the sweep
    (0.010, 1.00),
    (0.017, 1.00),
    (0.028, 1.00),
    (0.215, 1.00),
    (0.599, 1.00),
    (1.000, 1.00),
    # Vary λ in a MX row
    (0.010, 0.10),
    (0.010, 5.00),
    (0.010, 20.0),
    (0.215, 0.10),
    (0.215, 5.00),
    (0.215, 20.0),
]

print(f"{'mu':>7}  {'lam':>6}  {'delta_loc':>9}  {'h_ss':>7}  {'s_ss':>7}  (avg {len(SEEDS)} seeds)")
print("-" * 55)
for mu, lam in PROBE_MU_LAM:
    h_list, s_list = [], []
    for seed in SEEDS:
        h, s = run_sim(mu, lam, delta_loc=2000.0, seed=seed)
        h_list.append(h); s_list.append(s)
        sys.stdout.write(f"\r  running µ={mu:.3f} λ={lam:.2f} seed={seed}  ")
        sys.stdout.flush()
    print(f"\r{mu:>7.3f}  {lam:>6.2f}  {'2000':>9}  {np.mean(h_list):>7.3f}  {np.mean(s_list):>7.3f}")

# Now probe δ_loc variation at a low-µ MX cell to find GC
print()
print("--- δ_loc probe at µ=0.028, λ=1.0 ---")
for dloc in [500, 1000, 2000, 4000, 8000, 15000]:
    h_list, s_list = [], []
    for seed in SEEDS:
        h, s = run_sim(0.028, 1.0, delta_loc=dloc, seed=seed)
        h_list.append(h); s_list.append(s)
        sys.stdout.write(f"\r  δ_loc={dloc} seed={seed}  ")
        sys.stdout.flush()
    print(f"\r{'0.028':>7}  {'1.00':>6}  {dloc:>9}  {np.mean(h_list):>7.3f}  {np.mean(s_list):>7.3f}")

# Probe high µ for RTB
print()
print("--- high-µ probe at λ=0.1 for RTB ---")
for mu in [0.5, 1.0, 2.0, 5.0]:
    h_list, s_list = [], []
    for seed in SEEDS:
        h, s = run_sim(mu, 0.1, delta_loc=2000.0, seed=seed)
        h_list.append(h); s_list.append(s)
        sys.stdout.write(f"\r  µ={mu} seed={seed}  ")
        sys.stdout.flush()
    print(f"\r{mu:>7.3f}  {'0.10':>6}  {'2000':>9}  {np.mean(h_list):>7.3f}  {np.mean(s_list):>7.3f}")
