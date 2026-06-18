"""
diagnose_payoff.py
------------------
Find the (t, tau_BA) combinations that flip the static payoff gap
Π_S − Π_L from negative to positive at (h=0.5, s=0.5).

Also test: once GC is reachable (s→1), does the emission replicator
then push h→0 (L-firms dominate under universal tax)? That would give
a clean GC attractor.

And test: is RTB reachable at low t / low tau_BA? We already know it is,
but confirm the clean corner.
"""

import sys
import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, average_profits, emission_replicator, relocate_firms
from model.jurisdictions import (
    fiscal_revenues, payoff_matrix, network_correction, policy_replicator,
)

SEED = 42
rng0 = np.random.default_rng(SEED)
N    = 50
P0   = np.exp(rng0.normal(7.0, 1.5, size=N))
sig0 = (rng0.random(N) < 0.5).astype(int)
fl0  = rng0.integers(0, N, size=500)
ft0  = (rng0.random(500) < 0.5).astype(int)
G0, W0 = build_network(N, 3, "er", SEED)
k0     = effective_degree(G0)

def static_gap(t, tau_BA, tau=3.0, delta_loc=2000.0):
    p = Params(
        N=N, k=3, topology="er", M=500,
        c_H=4.0, c_L=6.0, F=0.0, a=20.0, b=1.0,
        t=t, tau=tau, g=2.4, tau_BA=tau_BA,
        delta_loc=delta_loc, delta_glob=500.0,
        mu=0.10, lam=2.0, kappa=1.0, relocate=False,
        dt=1.0, T=1, seed=SEED, mu_P=7.0, sigma_P=1.5, s0=0.5, h0=0.5,
    )
    fH, fL       = count_firms(fl0, ft0, N)
    ps, qH, qL   = solve_market(fH, fL, sig0, P0, W0, p)
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(fH, fL, sig0, P0, ps, W0, qH, qL, p)
    b_SL = network_correction(a_SS, a_SL, a_LS, a_LL, k0)
    s    = 0.5
    Pi_S = s * (a_SS + b_SL) + (1 - s) * (a_SL + b_SL)
    Pi_L = s * (a_LS - b_SL) + (1 - s) * (a_LL - b_SL)
    return Pi_S - Pi_L, a_SS, a_SL, a_LS, a_LL

# ── (A) Grid: t × tau_BA ────────────────────────────────────────────────────

T_VALS      = [1, 2, 3, 5, 8, 10, 15, 20]
TAU_BA_VALS = [0, 2, 5, 8, 10, 15, 20, 30]

print("=" * 72)
print("(A) Static Π_S−Π_L at (h=0.5, s=0.5)  as function of t and τ_BA")
print("    Positive = strict dominates = GC attractor")
print("=" * 72)
_col_label = "t\\τ_BA"
header = f"  {_col_label:>8} " + "".join(f"  {v:>6}" for v in TAU_BA_VALS)
print(header)
print("  " + "-" * (len(header)))
for t in T_VALS:
    row = f"  {t:>8} "
    for tba in TAU_BA_VALS:
        gap, *_ = static_gap(t, tba)
        marker = " *" if gap > 0 else "  "
        row += f"  {gap:>5.0f}{marker}"
    print(row)
print("  (* = positive gap → GC attractor)")

# ── (B) Full dynamics at promising (t, tau_BA) combos ───────────────────────

T_SIM  = 500
BURNIN = 100
SEEDS  = [42, 123, 7]

def run_full(t, tau_BA, delta_loc=2000.0, mu=0.10, lam=2.0, seed=SEED):
    p = Params(
        N=N, k=3, topology="er", M=500,
        c_H=4.0, c_L=6.0, F=0.0, a=20.0, b=1.0,
        t=t, tau=3.0, g=2.4, tau_BA=tau_BA,
        delta_loc=delta_loc, delta_glob=500.0,
        mu=mu, lam=lam, kappa=1.0, relocate=True,
        dt=1.0, T=T_SIM, seed=seed,
        mu_P=7.0, sigma_P=1.5, s0=0.5, h0=0.5,
    )
    rng       = np.random.default_rng(p.seed)
    P_pop     = np.exp(rng.normal(p.mu_P, p.sigma_P, size=N))
    sigma     = (rng.random(N) < p.s0).astype(int)
    firm_loc  = rng.integers(0, N, size=p.M)
    firm_type = (rng.random(p.M) < p.h0).astype(int)
    G, W      = build_network(N, p.k, p.topology, p.seed)
    k_eff     = effective_degree(G)

    h_ser = np.empty(T_SIM)
    s_ser = np.empty(T_SIM)

    for period in range(1, T_SIM + 1):
        f_H, f_L            = count_firms(firm_loc, firm_type, N)
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

        h_ser[period - 1] = float(np.mean(firm_type))
        s_ser[period - 1] = float(np.mean(sigma))

    return float(np.mean(h_ser[BURNIN:])), float(np.mean(s_ser[BURNIN:]))

print()
print("=" * 72)
print("(B) Full dynamics — candidate (t, τ_BA) combos for GC and PH")
print(f"    T={T_SIM}, burn-in={BURNIN}, {len(SEEDS)} seeds, µ=0.10, λ=2.0, relocate=True")
print("=" * 72)
print(f"  {'t':>4}  {'τ_BA':>6}  {'h_ss':>7}  {'s_ss':>7}  {'regime':>6}")
print("  " + "-" * 38)

CANDIDATES = [
    # (t,  tau_BA)  — chosen from cells with positive static gap
    ( 5,  10),
    ( 5,  15),
    ( 5,  20),
    ( 8,   5),
    ( 8,  10),
    (10,   5),
    (10,  10),
    (15,   2),
    (15,   5),
    (20,   0),
    (20,   2),
    # Low t for RTB confirmation
    ( 2,   0),
    ( 3,   0),
    # Intermediate for PH
    ( 5,   5),
    ( 8,   2),
    (10,   2),
]

def classify(h, s, thr=0.15):
    if h > 1-thr and s < thr:   return "RTB"
    if h < thr   and s > 1-thr: return "GC"
    if thr <= h <= 1-thr and thr <= s <= 1-thr: return "PH"
    return "MX"

for t, tba in CANDIDATES:
    hs, ss_list = [], []
    for seed in SEEDS:
        h, s = run_full(t, tba, seed=seed)
        hs.append(h); ss_list.append(s)
        sys.stdout.write(f"\r  t={t} τ_BA={tba} seed={seed}  ")
        sys.stdout.flush()
    h_ss = np.mean(hs); s_ss = np.mean(ss_list)
    reg  = classify(h_ss, s_ss)
    print(f"\r  {t:>4}  {tba:>6}  {h_ss:>7.3f}  {s_ss:>7.3f}  {reg:>6}")
