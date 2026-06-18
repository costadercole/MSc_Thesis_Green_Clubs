"""
diagnose_policy.py
------------------
Two diagnostics:

(A) Policy replicator trace — at what δ_loc does s stop collapsing to 0?
    Run T=200 for δ_loc in {2000, 5000, 10000, 20000, 50000} and print
    s(t) at t=1,2,5,10,25,50,100,200.

(B) Static payoff gap — print a_SS, a_SL, a_LS, a_LL and Π_S−Π_L
    at (h=0.5, s=0.5) for each δ_loc value.
    This tells us the sign of the policy replicator at the starting point.

(C) Also test: what happens if we fix h=1 (all H) and vary δ_loc?
    That's the absorbing state the firm replicator reaches — does the
    policy replicator ever push s away from 0 in that state?
"""

import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, per_capita_welfare, payoff_matrix,
    network_correction, policy_replicator,
)

T_SHORT = 300
SEED    = 42

def run_trace(delta_loc, h0=0.5, s0=0.5, relocate=True, seed=SEED, T=T_SHORT):
    p = Params(
        N=50, k=3, topology="er",
        M=500, c_H=4.0, c_L=6.0, F=0.0,
        a=20.0, b=1.0,
        t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
        delta_loc=delta_loc, delta_glob=500.0,
        mu=0.10, lam=2.0, kappa=1.0,
        relocate=relocate,
        dt=1.0, T=T, seed=seed,
        mu_P=7.0, sigma_P=1.5,
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

    return h_ser, s_ser

# ── (A) Policy trace: vary δ_loc ────────────────────────────────────────────

DLOC_VALS = [2000, 5000, 10000, 20000, 50000, 100000]
REPORT_T  = [1, 2, 3, 5, 10, 25, 50, 100, 200, 300]

print("=" * 70)
print("(A) s(t) trajectory for varying δ_loc  (µ=0.10, λ=2.0, relocate=True)")
print("=" * 70)
header = f"  {'δ_loc':>8}  " + "  ".join(f"t={t:>3}" for t in REPORT_T)
print(header)
print("  " + "-" * (len(header) - 2))

for dloc in DLOC_VALS:
    h_ser, s_ser = run_trace(dloc)
    vals = "  ".join(f"{s_ser[t-1]:>6.3f}" for t in REPORT_T)
    print(f"  {dloc:>8}  {vals}   final h={h_ser[-1]:.3f}")

# ── (B) Static payoff gap at (h=0.5, s=0.5) ────────────────────────────────

print()
print("=" * 70)
print("(B) Static payoff gap  Π_S − Π_L  at h=0.5, s=0.5  (no dynamics)")
print("=" * 70)
print(f"  {'δ_loc':>8}  {'a_SS':>8}  {'a_SL':>8}  {'a_LS':>8}  {'a_LL':>8}  {'ΠS-ΠL':>10}")
print("  " + "-" * 62)

rng0 = np.random.default_rng(SEED)
P0   = np.exp(rng0.normal(7.0, 1.5, size=50))
sig0 = (rng0.random(50) < 0.5).astype(int)
fl0  = rng0.integers(0, 50, size=500)
ft0  = (rng0.random(500) < 0.5).astype(int)
G0, W0 = build_network(50, 3, "er", SEED)
k0     = effective_degree(G0)

for dloc in DLOC_VALS:
    p0 = Params(
        N=50, k=3, topology="er", M=500,
        c_H=4.0, c_L=6.0, F=0.0, a=20.0, b=1.0,
        t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
        delta_loc=dloc, delta_glob=500.0,
        mu=0.10, lam=2.0, kappa=1.0, relocate=False,
        dt=1.0, T=1, seed=SEED, mu_P=7.0, sigma_P=1.5, s0=0.5, h0=0.5,
    )
    fH0, fL0       = count_firms(fl0, ft0, 50)
    pstar0, qH0, qL0 = solve_market(fH0, fL0, sig0, P0, W0, p0)
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(fH0, fL0, sig0, P0, pstar0, W0, qH0, qL0, p0)
    b_SL = network_correction(a_SS, a_SL, a_LS, a_LL, k0)
    s    = 0.5
    Pi_S = s * (a_SS + b_SL) + (1 - s) * (a_SL + b_SL)
    Pi_L = s * (a_LS - b_SL) + (1 - s) * (a_LL - b_SL)
    print(f"  {dloc:>8}  {a_SS:>8.1f}  {a_SL:>8.1f}  {a_LS:>8.1f}  {a_LL:>8.1f}  {Pi_S-Pi_L:>10.2f}")

# ── (C) Policy replicator at h=1, s=0.1 (late-time state) ──────────────────

print()
print("=" * 70)
print("(C) Can the policy replicator escape s→0 when h=1?")
print("    (start from h=1.0, s=0.1, vary δ_loc)")
print("=" * 70)
print(f"  {'δ_loc':>8}  " + "  ".join(f"t={t:>3}" for t in [1,2,5,10,25,50,100,200,300]))
print("  " + "-" * 62)

for dloc in DLOC_VALS:
    h_ser, s_ser = run_trace(dloc, h0=1.0, s0=0.1)
    vals = "  ".join(f"{s_ser[t-1]:>6.3f}" for t in [1,2,5,10,25,50,100,200,300])
    print(f"  {dloc:>8}  {vals}   final h={h_ser[-1]:.3f}")

# ── (D) What if relocate=False? Does that let s stabilise? ─────────────────

print()
print("=" * 70)
print("(D) relocate=False — does s stabilise at interior δ_loc values?")
print("=" * 70)
print(f"  {'δ_loc':>8}  " + "  ".join(f"t={t:>3}" for t in REPORT_T))
print("  " + "-" * 62)

for dloc in [2000, 5000, 10000, 20000]:
    h_ser, s_ser = run_trace(dloc, relocate=False)
    vals = "  ".join(f"{s_ser[t-1]:>6.3f}" for t in REPORT_T)
    print(f"  {dloc:>8}  {vals}   final h={h_ser[-1]:.3f}  s={s_ser[-1]:.3f}")
