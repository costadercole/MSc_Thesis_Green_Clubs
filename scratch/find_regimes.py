"""
find_regimes.py  (v3)
---------------------
Demonstrate that firm mobility µ is the key lever determining which of the
three coevolutionary regimes the economy converges to.

Starting point: h0=0.4, s0=0.6  (away from the basin boundary at 0.5/0.5)
All other parameters: calibrated baseline throughout.

Step 1 — µ sweep
  Run T=1000 for µ in a log-spaced grid [0.001, 2.0], 5 seeds each.
  Print h_ss and s_ss per µ value and classify regime.

Step 2 — Trajectory plots
  For the best µ value in each regime (RTB / GC / PH), run T=1000 seed=42
  and save a labelled h(t), s(t) plot.

Outputs
-------
output/mu_sweep_table.txt     — printed regime table
output/regime_RTB.png
output/regime_GC.png
output/regime_PH.png
output/regime_mu_summary.png  — single figure with all three trajectories
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import Params
from model.network import build_network, effective_degree
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, payoff_matrix, network_correction, policy_replicator,
)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

T_SIM   = 1000
BURNIN  = 200
SEEDS   = [42, 123, 7, 999, 2024]

# Starting point — away from the knife-edge basin boundary
H0 = 0.4
S0 = 0.6

# µ grid — log-spaced, 20 values from very low to very high mobility
MU_GRID = np.logspace(np.log10(0.001), np.log10(2.0), 20)

CORNER = 0.15

def classify(h, s):
    if h > 1 - CORNER and s < CORNER:                             return "RTB"
    if h < CORNER     and s > 1 - CORNER:                         return "GC"
    if CORNER <= h <= 1-CORNER and CORNER <= s <= 1-CORNER:       return "PH"
    return "MX"

os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(mu, h0=H0, s0=S0, seed=42, T=T_SIM):
    p = Params(
        N=50, k=3, topology="er",
        M=500, c_H=4.0, c_L=6.0, F=0.0,
        a=20.0, b=1.0,
        t=5.0, tau=3.0, g=2.4, tau_BA=0.0,
        delta_loc=2000.0, delta_glob=500.0,
        mu=mu, lam=2.0, kappa=1.0,
        relocate=True,
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

    h_ss = float(np.mean(h_ser[BURNIN:]))
    s_ss = float(np.mean(s_ser[BURNIN:]))
    return h_ser, s_ser, h_ss, s_ss

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — µ sweep
# ─────────────────────────────────────────────────────────────────────────────

total = len(MU_GRID) * len(SEEDS)
print("=" * 68)
print(f"  µ SWEEP  —  h0={H0}, s0={S0}, T={T_SIM}, burn-in={BURNIN}")
print(f"  {len(MU_GRID)} µ values × {len(SEEDS)} seeds = {total} simulations")
print("=" * 68)
print(f"  {'µ':>8}  {'h_ss':>7}  {'s_ss':>7}  {'regime':>6}  {'h_std':>7}  {'s_std':>7}")
print("  " + "-" * 52)

mu_results  = {}   # mu -> (h_ss_mean, s_ss_mean, regime, h_ser_s42, s_ser_s42)
run_idx = 0
t0 = time.time()

for mu in MU_GRID:
    h_list, s_list = [], []
    h_ser_42 = s_ser_42 = None

    for seed in SEEDS:
        run_idx += 1
        elapsed = time.time() - t0
        eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
        sys.stdout.write(
            f"\r  [{run_idx:>3}/{total}]  µ={mu:.4f}  seed={seed}"
            f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s    "
        )
        sys.stdout.flush()

        h_ser, s_ser, h_ss, s_ss = run_sim(mu, seed=seed)
        h_list.append(h_ss)
        s_list.append(s_ss)
        if seed == 42:
            h_ser_42 = h_ser
            s_ser_42 = s_ser

    h_m   = float(np.mean(h_list))
    s_m   = float(np.mean(s_list))
    h_std = float(np.std(h_list))
    s_std = float(np.std(s_list))
    reg   = classify(h_m, s_m)
    mu_results[mu] = (h_m, s_m, reg, h_ser_42, s_ser_42, h_std, s_std)
    print(f"\r  {mu:>8.4f}  {h_m:>7.3f}  {s_m:>7.3f}  {reg:>6}  {h_std:>7.3f}  {s_std:>7.3f}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Pick best representative µ per regime
# ─────────────────────────────────────────────────────────────────────────────

best = {}
for mu, (h_m, s_m, reg, h_ser, s_ser, h_std, s_std) in mu_results.items():
    if reg == "RTB": score = h_m - s_m - (h_std + s_std)
    elif reg == "GC": score = s_m - h_m - (h_std + s_std)
    elif reg == "PH": score = -(abs(h_m-0.5) + abs(s_m-0.5)) - (h_std + s_std)
    else: continue
    if reg not in best or score > best[reg]["score"]:
        best[reg] = dict(mu=mu, h_ss=h_m, s_ss=s_m, score=score,
                         h_ser=h_ser, s_ser=s_ser)

print("=" * 68)
print("  REGIME SUMMARY  (best µ per regime, h0=0.4, s0=0.6)")
print("=" * 68)
RNAMES = {
    "RTB": "Race to the Bottom  (h→1, s→0)",
    "GC":  "Green Club          (h→0, s→1)",
    "PH":  "Persistent Heterog. (h,s interior)",
}
for key in ["GC", "PH", "RTB"]:
    if key in best:
        b = best[key]
        print(f"  {RNAMES[key]}")
        print(f"    µ = {b['mu']:.4f}  →  h_ss={b['h_ss']:.3f}  s_ss={b['s_ss']:.3f}")
    else:
        print(f"  {RNAMES[key]}  —  NOT FOUND")
print("=" * 68)

# Save summary table
with open("output/mu_sweep_table.txt", "w") as f:
    f.write(f"µ sweep  h0={H0} s0={S0} T={T_SIM} burn-in={BURNIN}\n")
    f.write(f"{'mu':>8}  {'h_ss':>7}  {'s_ss':>7}  {'regime':>6}\n")
    for mu, (h_m, s_m, reg, *_) in mu_results.items():
        f.write(f"{mu:>8.4f}  {h_m:>7.3f}  {s_m:>7.3f}  {reg:>6}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Individual trajectory plots
# ─────────────────────────────────────────────────────────────────────────────

TRAJ_COL = {
    "RTB": ("#d62728", "#ff7f0e"),
    "GC":  ("#2ca02c", "#17becf"),
    "PH":  ("#9467bd", "#8c564b"),
}
TRAJ_NAMES = {
    "RTB": "Race to the Bottom",
    "GC":  "Green Club",
    "PH":  "Persistent Heterogeneity",
}

periods = np.arange(1, T_SIM + 1)

for key in ["RTB", "GC", "PH"]:
    if key not in best:
        print(f"  [SKIP] {key} not found")
        continue

    b = best[key]
    col_h, col_s = TRAJ_COL[key]
    h_ser, s_ser = b["h_ser"], b["s_ser"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_h, ax_s = axes

    ax_h.plot(periods, h_ser, color=col_h, linewidth=1.2, label="h(t)  H-firm share")
    ax_h.axvline(BURNIN, color="black", linestyle=":", linewidth=1.0,
                 label=f"burn-in = {BURNIN}")
    ax_h.axhline(b["h_ss"], color=col_h, linestyle="--", linewidth=0.9, alpha=0.8,
                 label=f"h_ss = {b['h_ss']:.3f}")
    ax_h.set_ylabel("h(t)", fontsize=11)
    ax_h.set_ylim(-0.05, 1.05)
    ax_h.legend(fontsize=9, loc="upper right")
    ax_h.grid(True, alpha=0.3)

    ax_s.plot(periods, s_ser, color=col_s, linewidth=1.2,
              label="s(t)  strict-jur. share")
    ax_s.axvline(BURNIN, color="black", linestyle=":", linewidth=1.0,
                 label=f"burn-in = {BURNIN}")
    ax_s.axhline(b["s_ss"], color=col_s, linestyle="--", linewidth=0.9, alpha=0.8,
                 label=f"s_ss = {b['s_ss']:.3f}")
    ax_s.set_ylabel("s(t)", fontsize=11)
    ax_s.set_xlabel("Period", fontsize=11)
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.legend(fontsize=9, loc="upper right")
    ax_s.grid(True, alpha=0.3)

    fig.suptitle(
        f"Regime: {TRAJ_NAMES[key]}\n"
        f"µ = {b['mu']:.4f}  |  h₀={H0}  s₀={S0}  seed=42  "
        f"T={T_SIM}  burn-in={BURNIN}  relocate=True",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join("output", f"regime_{key.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Summary figure: all three trajectories side by side
# ─────────────────────────────────────────────────────────────────────────────

found_keys = [k for k in ["GC", "PH", "RTB"] if k in best]

if found_keys:
    fig, axes = plt.subplots(2, len(found_keys),
                             figsize=(5 * len(found_keys), 7), sharex=True)
    if len(found_keys) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col_idx, key in enumerate(found_keys):
        b = best[key]
        col_h, col_s = TRAJ_COL[key]
        ax_h = axes[0, col_idx]
        ax_s = axes[1, col_idx]

        ax_h.plot(periods, b["h_ser"], color=col_h, linewidth=1.0)
        ax_h.axhline(b["h_ss"], color=col_h, linestyle="--", linewidth=0.8, alpha=0.7)
        ax_h.axvline(BURNIN, color="black", linestyle=":", linewidth=0.8)
        ax_h.set_ylim(-0.05, 1.05)
        ax_h.set_title(f"{TRAJ_NAMES[key]}\nµ={b['mu']:.4f}", fontsize=10)
        ax_h.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_h.set_ylabel("h(t)  H-firm share", fontsize=10)

        ax_s.plot(periods, b["s_ser"], color=col_s, linewidth=1.0)
        ax_s.axhline(b["s_ss"], color=col_s, linestyle="--", linewidth=0.8, alpha=0.7)
        ax_s.axvline(BURNIN, color="black", linestyle=":", linewidth=0.8)
        ax_s.set_ylim(-0.05, 1.05)
        ax_s.set_xlabel("Period", fontsize=10)
        ax_s.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_s.set_ylabel("s(t)  strict-jur. share", fontsize=10)

    fig.suptitle(
        f"Three coevolutionary regimes  —  h₀={H0}, s₀={S0}, baseline parameters\n"
        f"T={T_SIM}, burn-in={BURNIN}, seed=42, relocate=True",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join("output", "regime_mu_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
