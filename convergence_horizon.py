"""
convergence_horizon.py
----------------------
Run the full coevolutionary model for 2000 periods at a single moderate
parameter set (calibrated baseline with relocation enabled, µ=0.10, λ=2.0)
and diagnose when h(t) and s(t) stabilise.

Outputs
-------
output/convergence_horizon.png   — time-series plot of h and s
Printed to stdout:
  - rolling-window std dev used to detect stabilisation
  - recommended T and burn-in for subsequent runs
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, per_capita_welfare, payoff_matrix,
    network_correction, policy_replicator,
)

# ─────────────────────────────────────────────────────────────────────────────
# Parameter set — calibrated baseline, relocation on, moderate µ
# ─────────────────────────────────────────────────────────────────────────────

T_LONG = 2000   # long horizon to diagnose convergence

p = Params(
    # network
    N=50, k=3, topology="er",
    # firms / market
    M=500, c_H=4.0, c_L=6.0, F=0.0,
    a=20.0, b=1.0,
    # policy
    t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
    # damage
    delta_loc=2000.0, delta_glob=500.0,
    # dynamics  — moderate µ that previously showed non-convergence at 150 periods
    mu=0.10, lam=2.0, kappa=1.0,
    relocate=True,
    # simulation
    dt=1.0, T=T_LONG, seed=42,
    mu_P=7.0, sigma_P=1.5,
    s0=0.5, h0=0.5,
)

# ─────────────────────────────────────────────────────────────────────────────
# Initialise state
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(p.seed)

P_pop = np.exp(rng.normal(p.mu_P, p.sigma_P, size=p.N))
sigma = (rng.random(p.N) < p.s0).astype(int)
locs  = rng.integers(0, p.N, size=p.M)
types = (rng.random(p.M) < p.h0).astype(int)
firm_loc  = locs.copy()
firm_type = types.copy()

G, W  = build_network(p.N, p.k, p.topology, p.seed)
k_eff = effective_degree(G)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

h_series = np.empty(T_LONG)
s_series = np.empty(T_LONG)

def _progress(period, total, width=50):
    filled = int(width * period / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * period / total
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  t={period}/{total}")
    sys.stdout.flush()

print(f"Running convergence horizon: T={T_LONG}, µ={p.mu}, λ={p.lam}, relocate={p.relocate}")

for period in range(1, T_LONG + 1):

    f_H, f_L = count_firms(firm_loc, firm_type, p.N)
    p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P_pop, W, p)

    TR         = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
    welfare_pc = per_capita_welfare(f_H, sigma, P_pop, p_star, TR, p)

    pi_H, pi_L         = firm_variable_profits(q_H, q_L, P_pop, p)
    pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)

    # Relocation
    firm_loc = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W)

    # Emission replicator
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

    # Policy replicator
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P_pop, p_star, W, q_H, q_L, p)
    b_SL  = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
    s_old = float(np.mean(sigma))
    s_new = policy_replicator(s_old, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt, p.kappa)

    gap_s    = s_new - s_old
    n_change = int(round(abs(gap_s) * p.N))
    if n_change > 0:
        if gap_s > 0:
            candidates = np.where(sigma == 0)[0]
            n = min(n_change, len(candidates))
            if n > 0:
                sigma[rng.choice(candidates, n, replace=False)] = 1
        else:
            candidates = np.where(sigma == 1)[0]
            n = min(n_change, len(candidates))
            if n > 0:
                sigma[rng.choice(candidates, n, replace=False)] = 0

    h_series[period - 1] = float(np.mean(firm_type))
    s_series[period - 1] = float(np.mean(sigma))

    _progress(period, T_LONG)

print()  # newline

# ─────────────────────────────────────────────────────────────────────────────
# Stabilisation detection — rolling std dev in a 100-period window
# ─────────────────────────────────────────────────────────────────────────────

WINDOW     = 100
STABLE_THR = 0.01   # rolling std < this threshold → "stable"

t_stab_h = T_LONG   # default: never stabilises
t_stab_s = T_LONG

for t in range(WINDOW, T_LONG):
    w_h = h_series[t - WINDOW : t]
    w_s = s_series[t - WINDOW : t]
    if np.std(w_h) < STABLE_THR and t_stab_h == T_LONG:
        t_stab_h = t
    if np.std(w_s) < STABLE_THR and t_stab_s == T_LONG:
        t_stab_s = t

t_stab = max(t_stab_h, t_stab_s)   # both must be stable

# Recommended T: t_stab + at least 500 more periods to average over,
# rounded up to nearest 500, with a minimum of 1000.
T_rec    = max(1000, int(np.ceil((t_stab + 500) / 500) * 500))
burnin   = int(np.ceil(t_stab / 100) * 100)   # round up to nearest 100

# ─────────────────────────────────────────────────────────────────────────────
# Print report
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 64
print(f"\n{SEP}")
print("  CONVERGENCE HORIZON REPORT")
print(SEP)
print(f"  Horizon run           : {T_LONG} periods")
print(f"  Stabilisation window  : {WINDOW} periods  (threshold σ < {STABLE_THR})")
print(f"  h(t) stabilises at t ≈ {t_stab_h}")
print(f"  s(t) stabilises at t ≈ {t_stab_s}")
print(f"  Joint stable from t   : {t_stab}")
print(SEP)
print(f"  >>> Recommended T     : {T_rec} periods")
print(f"  >>> Recommended burn-in: {burnin} periods")
print(SEP)
print(f"  Final h (last 200 periods): {h_series[-200:].mean():.4f} ± {h_series[-200:].std():.4f}")
print(f"  Final s (last 200 periods): {s_series[-200:].mean():.4f} ± {s_series[-200:].std():.4f}")
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

periods = np.arange(1, T_LONG + 1)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax_h, ax_s = axes

ax_h.plot(periods, h_series, color="#d62728", linewidth=1.0, label="h(t)  share of H-firms")
if t_stab_h < T_LONG:
    ax_h.axvline(t_stab_h, color="#d62728", linestyle="--", alpha=0.6,
                 label=f"h stable ≈ t={t_stab_h}")
ax_h.axvline(burnin, color="black", linestyle=":", linewidth=1.2,
             label=f"Burn-in = {burnin}")
ax_h.set_ylabel("h(t)  — H-firm share", fontsize=11)
ax_h.set_ylim(-0.05, 1.05)
ax_h.legend(fontsize=9, loc="upper right")
ax_h.grid(True, alpha=0.3)

ax_s.plot(periods, s_series, color="#1f77b4", linewidth=1.0, label="s(t)  share of strict jurisdictions")
if t_stab_s < T_LONG:
    ax_s.axvline(t_stab_s, color="#1f77b4", linestyle="--", alpha=0.6,
                 label=f"s stable ≈ t={t_stab_s}")
ax_s.axvline(burnin, color="black", linestyle=":", linewidth=1.2,
             label=f"Burn-in = {burnin}")
ax_s.set_ylabel("s(t)  — strict-jur. share", fontsize=11)
ax_s.set_xlabel("Period", fontsize=11)
ax_s.set_ylim(-0.05, 1.05)
ax_s.legend(fontsize=9, loc="upper right")
ax_s.grid(True, alpha=0.3)

title = (
    f"Convergence Horizon  —  T={T_LONG},  µ={p.mu},  λ={p.lam},  "
    f"κ={p.kappa},  relocate=True\n"
    f"Recommended T={T_rec},  burn-in={burnin}"
)
fig.suptitle(title, fontsize=11, y=1.01)
fig.tight_layout()

os.makedirs("output", exist_ok=True)
out_path = os.path.join("output", "convergence_horizon.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved to {out_path}")
