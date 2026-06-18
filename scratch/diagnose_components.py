"""
diagnose_components.py
----------------------
Decompose the payoff matrix into its components at (h=0.5, s=0.5) to find
exactly why W_L_base >> W_S_base, and what parameter changes close the gap.

Also tests: what initial condition (h0, s0) leads to different attractors?
The static table showed h_ss always 0.667 — is there a basin boundary?
"""

import numpy as np
from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, average_profits, emission_replicator
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

def decompose(t=5.0, tau_BA=0.0, tau=3.0, delta_loc=2000.0, c_H=4.0, c_L=6.0):
    p = Params(
        N=N, k=3, topology="er", M=500,
        c_H=c_H, c_L=c_L, F=0.0, a=20.0, b=1.0,
        t=t, tau=tau, g=2.4, tau_BA=tau_BA,
        delta_loc=delta_loc, delta_glob=500.0,
        mu=0.10, lam=2.0, kappa=1.0, relocate=False,
        dt=1.0, T=1, seed=SEED, mu_P=7.0, sigma_P=1.5, s0=0.5, h0=0.5,
    )
    fH, fL         = count_firms(fl0, ft0, N)
    ps, qH, qL     = solve_market(fH, fL, sig0, P0, W0, p)
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(fH, fL, sig0, P0, ps, W0, qH, qL, p)
    b_SL = network_correction(a_SS, a_SL, a_LS, a_LL, k0)

    strict_mask = sig0 == 1
    lax_mask    = sig0 == 0
    P_safe      = np.maximum(P0, 1.0)

    p_S = float(np.mean(ps[strict_mask]))
    p_L = float(np.mean(ps[lax_mask]))
    cs_S = (p.a - p_S)**2 / (2*p.b)
    cs_L = (p.a - p_L)**2 / (2*p.b)
    fH_S = float(np.mean(fH[strict_mask]))
    fH_L = float(np.mean(fH[lax_mask]))
    P_S  = float(np.mean(P0[strict_mask]))
    P_L  = float(np.mean(P0[lax_mask]))
    loc_dmg_S = delta_loc * fH_S / P_S
    loc_dmg_L = delta_loc * fH_L / P_L
    glob_dmg  = 500.0 * fH.sum() / P_safe.sum()

    s = 0.5
    Pi_S = s * (a_SS + b_SL) + (1-s) * (a_SL + b_SL)
    Pi_L = s * (a_LS - b_SL) + (1-s) * (a_LL - b_SL)

    return dict(
        p_S=p_S, p_L=p_L, cs_S=cs_S, cs_L=cs_L,
        loc_S=loc_dmg_S, loc_L=loc_dmg_L, glob=glob_dmg,
        a_SS=a_SS, a_SL=a_SL, a_LS=a_LS, a_LL=a_LL,
        b_SL=b_SL,
        Pi_S=Pi_S, Pi_L=Pi_L, gap=Pi_S-Pi_L,
    )

# ── (A) Baseline decomposition ───────────────────────────────────────────────

print("=" * 70)
print("(A) Payoff decomposition at baseline (t=5, τ_BA=0, δ_loc=2000)")
print("=" * 70)
d = decompose()
print(f"  Price:          p_S={d['p_S']:.2f}   p_L={d['p_L']:.2f}   (lax cheaper by {d['p_L']-d['p_S']:.2f})")
print(f"  CS per capita:  CS_S={d['cs_S']:.2f}   CS_L={d['cs_L']:.2f}   (lax higher by {d['cs_L']-d['cs_S']:.2f})")
print(f"  Local damage:   D_S={d['loc_S']:.2f}   D_L={d['loc_L']:.2f}")
print(f"  Global damage:  {d['glob']:.2f}  (same for all)")
print(f"  Payoffs:  a_SS={d['a_SS']:.2f}  a_SL={d['a_SL']:.2f}  a_LS={d['a_LS']:.2f}  a_LL={d['a_LL']:.2f}")
print(f"  b_SL={d['b_SL']:.4f}")
print(f"  Π_S={d['Pi_S']:.2f}   Π_L={d['Pi_L']:.2f}   gap={d['gap']:.2f}")

# ── (B) What closes the CS gap? Vary c_H (make H-firms less cheap) ──────────

print()
print("=" * 70)
print("(B) Vary c_H to reduce H-firm cost advantage  (t=5, τ_BA=0)")
print("    c_L fixed at 6.  H-firm advantage disappears when c_H→c_L.")
print("=" * 70)
print(f"  {'c_H':>5}  {'p_S':>6}  {'p_L':>6}  {'ΔCS':>7}  {'ΔΠ(gap)':>10}")
for ch in [4.0, 4.5, 5.0, 5.5, 5.8, 5.9, 6.0, 6.5, 7.0, 8.0]:
    d = decompose(c_H=ch)
    print(f"  {ch:>5.1f}  {d['p_S']:>6.2f}  {d['p_L']:>6.2f}  {d['cs_L']-d['cs_S']:>7.2f}  {d['gap']:>10.2f}")

# ── (C) Vary t with c_H=5.5 (near parity) ───────────────────────────────────

print()
print("=" * 70)
print("(C) Vary t with c_H=5.5  (H/L cost near-parity)")
print("=" * 70)
print(f"  {'t':>5}  {'τ_BA':>6}  {'gap':>10}  {'a_SS':>7}  {'a_LL':>7}")
for t in [1, 2, 3, 5, 8, 10, 15]:
    for tba in [0, 5, 10]:
        d = decompose(t=t, tau_BA=tba, c_H=5.5)
        marker = " *" if d['gap'] > 0 else ""
        print(f"  {t:>5}  {tba:>6}  {d['gap']:>10.2f}{marker}  {d['a_SS']:>7.2f}  {d['a_LL']:>7.2f}")

# ── (D) Basin of attraction: vary (h0, s0) at baseline ───────────────────────

print()
print("=" * 70)
print("(D) Basin sweep: vary initial (h0, s0) — which attractor does each reach?")
print("    (t=5, τ_BA=0, δ_loc=2000, µ=0.10, λ=2.0, T=300, seed=42)")
print("=" * 70)

import sys
from model.firms import relocate_firms

T_BASIN = 300

def run_basin(h0, s0):
    p = Params(
        N=N, k=3, topology="er", M=500,
        c_H=4.0, c_L=6.0, F=0.0, a=20.0, b=1.0,
        t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
        delta_loc=2000.0, delta_glob=500.0,
        mu=0.10, lam=2.0, kappa=1.0, relocate=True,
        dt=1.0, T=T_BASIN, seed=SEED,
        mu_P=7.0, sigma_P=1.5, s0=s0, h0=h0,
    )
    rng       = np.random.default_rng(p.seed)
    P_pop     = np.exp(rng.normal(p.mu_P, p.sigma_P, size=N))
    sigma     = (rng.random(N) < p.s0).astype(int)
    firm_loc  = rng.integers(0, N, size=p.M)
    firm_type = (rng.random(p.M) < p.h0).astype(int)
    G, W      = build_network(N, p.k, p.topology, p.seed)
    k_eff     = effective_degree(G)

    for period in range(1, T_BASIN + 1):
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

    return float(np.mean(firm_type)), float(np.mean(sigma))

H0_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]
S0_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]
_row_label = "h0\\s0"
print(f"  {_row_label:>6} " + "".join(f"  s0={s:>3.1f}" for s in S0_VALS))
for h0 in H0_VALS:
    row = f"  h0={h0:.1f} "
    for s0 in S0_VALS:
        sys.stdout.write(f"\r  running h0={h0} s0={s0}  ")
        sys.stdout.flush()
        h_f, s_f = run_basin(h0, s0)
        row += f"  ({h_f:.2f},{s_f:.2f})"
    print(f"\r{row}")
