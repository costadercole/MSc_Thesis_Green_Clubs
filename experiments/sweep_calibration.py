"""
Fine-grained calibration sweep around the interior-equilibrium region.

Targets:
  Carbon leakage rate   :  5 – 20 %
  Import penetration    : 20 – 40 %
  Strict/lax price gap  : 10 – 30 %

Run:  python experiments/sweep_calibration.py
Output: output/sweep_calibration.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import itertools
import numpy as np
import params as cfg
from params import Params
from model.network import build_network, effective_degree
from model.market import firm_variable_profits
from model.labour import solve_wages
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    fiscal_revenues, per_capita_welfare, payoff_matrix,
    network_correction, policy_replicator,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid
# ─────────────────────────────────────────────────────────────────────────────

DELTA_VALUES = [7, 8, 9, 10, 11, 12, 13, 14]
TAU_VALUES   = [10, 11, 12, 13, 14, 15, 16, 18, 20]
MU_VALUES    = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
T_VALUES     = [4.0, 5.0, 6.0, 7.0]

T_SIM   = 500
SS_FRAC = 0.75

TARGET_LEAKAGE   = (5,  20)
TARGET_IMPORT    = (20, 40)
TARGET_PRICE_GAP = (10, 30)

# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_once(delta, tau, mu, t):
    N = cfg.N
    M = len(cfg.firms_init)

    firm_loc  = np.array([f[0] for f in cfg.firms_init])
    firm_type = np.array([1 if f[1] == "H" else 0 for f in cfg.firms_init])
    sigma     = np.array(cfg.sigma_init, dtype=int)
    P         = np.array(cfg.P_init, dtype=float)

    G, W  = build_network(N, cfg.k, cfg.topology, cfg.seed)
    k_eff = effective_degree(G)

    p = Params(
        N=N, M=M, k=cfg.k, topology=cfg.topology,
        a=cfg.a, b=cfg.b, c_H=cfg.c_H, c_L=cfg.c_L, t=t,
        tau=tau, c_trade=cfg.c_trade, tau_BA=cfg.tau_BA,
        delta=delta, F=cfg.F, w_bar=cfg.w_bar, alpha=cfg.alpha,
        mu=mu, lam=cfg.lam,
        dt=cfg.dt, T=T_SIM, seed=cfg.seed,
        mu_P=cfg.mu_P, sigma_P=cfg.sigma_P,
        s0=float(np.mean(sigma)), h0=float(np.mean(firm_type)),
    )

    rng = np.random.default_rng(cfg.seed)

    strict_H_periods = 0.0
    leakage_moves    = 0
    ss_start         = int(T_SIM * SS_FRAC)
    import_shares    = []
    price_gaps       = []
    h_vals, s_vals   = [], []

    for period in range(1, T_SIM + 1):
        f_H, f_L = count_firms(firm_loc, firm_type, N)
        wages, p_star, q_H, q_L = solve_wages(f_H, f_L, sigma, P, W, p)
        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        pi_H, pi_L         = firm_variable_profits(q_H, q_L, P, p)
        pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)

        strict_H_periods += float(f_H[sigma == 1].sum())

        old_loc  = firm_loc.copy()
        firm_loc = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W)
        for m in range(M):
            if firm_loc[m] != old_loc[m]:
                if sigma[old_loc[m]] == 1 and sigma[firm_loc[m]] == 0:
                    leakage_moves += 1

        h_old = float(np.mean(firm_type))
        h_new = emission_replicator(h_old, pi_H_bar, pi_L_bar, p.dt)
        gap   = h_new - h_old
        if abs(gap) > 0.5 / M:
            if gap > 0:
                idx = np.where(firm_type == 0)[0]
                n = min(int(round(gap * M)), len(idx))
                if n > 0:
                    firm_type[rng.choice(idx, n, replace=False)] = 1
            else:
                idx = np.where(firm_type == 1)[0]
                n = min(int(round(-gap * M)), len(idx))
                if n > 0:
                    firm_type[rng.choice(idx, n, replace=False)] = 0

        a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P, p_star, W, wages, q_H, q_L, p)
        b_SL  = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
        s_old = float(np.mean(sigma))
        s_new = policy_replicator(s_old, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt)
        gap_s = s_new - s_old
        n_ch  = int(round(abs(gap_s) * N))
        if n_ch > 0:
            if gap_s > 0:
                cands = np.where(sigma == 0)[0]
                sigma[rng.choice(cands, min(n_ch, len(cands)), replace=False)] = 1
            else:
                cands = np.where(sigma == 1)[0]
                sigma[rng.choice(cands, min(n_ch, len(cands)), replace=False)] = 0

        smask, lmask = sigma == 1, sigma == 0
        if smask.any() and lmask.any():
            p_s = float(np.mean(p_star[smask]))
            p_l = float(np.mean(p_star[lmask]))
            if p_l > 0:
                price_gaps.append((p_s - p_l) / p_l * 100)

        if period > ss_start:
            imp_num = imp_den = 0.0
            for ii in range(N):
                imp_den += f_H[ii] * float(q_H[ii, ii]) + f_L[ii] * float(q_L[ii, ii])
                for jj in range(N):
                    if jj != ii and W[ii, jj] > 0:
                        imp_q = f_H[jj] * float(q_H[ii, jj]) + f_L[jj] * float(q_L[ii, jj])
                        imp_num += imp_q
                        imp_den += imp_q
            if imp_den > 0:
                import_shares.append(imp_num / imp_den * 100)
            h_vals.append(float(np.mean(firm_type)))
            s_vals.append(float(np.mean(sigma)))

    leakage    = (leakage_moves / strict_H_periods * 100) if strict_H_periods > 0 else 0.0
    import_pen = float(np.mean(import_shares)) if import_shares else float("nan")
    price_gap  = float(np.mean(price_gaps))    if price_gaps    else float("nan")
    h_ss       = float(np.mean(h_vals))        if h_vals        else float("nan")
    s_ss       = float(np.mean(s_vals))        if s_vals        else float("nan")

    def _in(v, lo, hi): return lo <= v <= hi

    passes = (
        _in(leakage,    *TARGET_LEAKAGE)   and
        _in(import_pen, *TARGET_IMPORT)    and
        _in(price_gap,  *TARGET_PRICE_GAP)
    )
    return {
        "delta": delta, "tau": tau, "mu": mu, "t": t,
        "leakage_pct":    round(leakage,    2),
        "import_pen_pct": round(import_pen, 2),
        "price_gap_pct":  round(price_gap,  2),
        "h_ss":           round(h_ss,       4),
        "s_ss":           round(s_ss,       4),
        "leakage_ok":     int(_in(leakage,    *TARGET_LEAKAGE)),
        "import_ok":      int(_in(import_pen, *TARGET_IMPORT)),
        "price_gap_ok":   int(_in(price_gap,  *TARGET_PRICE_GAP)),
        "all_pass":       int(passes),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Run sweep
# ─────────────────────────────────────────────────────────────────────────────

grid   = list(itertools.product(DELTA_VALUES, TAU_VALUES, MU_VALUES, T_VALUES))
n_runs = len(grid)
results = []

print(f"Calibration sweep: {len(DELTA_VALUES)}δ × {len(TAU_VALUES)}τ × {len(MU_VALUES)}μ × {len(T_VALUES)}t"
      f" = {n_runs} runs  (T={T_SIM} each,  c_trade={cfg.c_trade} fixed)")
print(f"{'δ':>5} {'τ':>5} {'μ':>5} {'t':>5}  {'leak%':>6} {'imp%':>6} {'pgap%':>6}  pass")
print("─" * 58)

for i, (delta, tau, mu, t) in enumerate(grid):
    r = run_once(delta, tau, mu, t)
    results.append(r)
    status = "✓" if r["all_pass"] else " "
    print(f"  [{status}] δ={delta:4.0f} τ={tau:3.0f} μ={mu:.2f} t={t:.0f}"
          f"  {r['leakage_pct']:6.1f} {r['import_pen_pct']:6.1f} {r['price_gap_pct']:6.1f}"
          f"  s_ss={r['s_ss']:.2f}")
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)
out_path = "output/sweep_calibration.csv"
fieldnames = [
    "delta", "tau", "mu", "t",
    "leakage_pct", "import_pen_pct", "price_gap_pct",
    "h_ss", "s_ss",
    "leakage_ok", "import_ok", "price_gap_ok", "all_pass",
]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

passing = [r for r in results if r["all_pass"]]
print(f"\n{'─'*58}")
print(f"  {len(passing)} / {n_runs} combinations pass all three targets.")
if passing:
    for r in passing:
        print(f"    δ={r['delta']} τ={r['tau']} μ={r['mu']} t={r['t']}"
              f"  leak={r['leakage_pct']}%  imp={r['import_pen_pct']}%"
              f"  pgap={r['price_gap_pct']}%  s_ss={r['s_ss']}")
print(f"\n  Saved: {out_path}")
