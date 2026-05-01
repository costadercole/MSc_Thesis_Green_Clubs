"""
Calibration sweep: find (c_trade, t) combinations that satisfy all three
empirical validation targets simultaneously.

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
from model.market import equilibrium_prices_with_trade, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import tariff_payoffs, payoff_matrix, network_correction, policy_replicator

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid  —  edit these to change the search space
# ─────────────────────────────────────────────────────────────────────────────

C_TRADE_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
T_VALUES       = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

T_SIM     = 200   # periods per run (shorter than full run for speed)
SS_FRAC   = 0.8   # last (1 - SS_FRAC) fraction used as steady-state window

# Validation targets [lo, hi]
TARGET_LEAKAGE   = (5,  20)
TARGET_IMPORT    = (20, 40)
TARGET_PRICE_GAP = (10, 30)

# ─────────────────────────────────────────────────────────────────────────────
# Core simulation — returns validation moments for one (c_trade, t) pair
# ─────────────────────────────────────────────────────────────────────────────

def run_once(c_trade, t):
    N = cfg.N
    M = len(cfg.firms_init)

    firm_loc  = np.array([f[0] for f in cfg.firms_init])
    firm_type = np.array([1 if f[1] == "H" else 0 for f in cfg.firms_init])
    sigma     = np.array(cfg.sigma_init, dtype=int)
    P         = np.array(cfg.P_init, dtype=float)

    G, W  = build_network(N, cfg.k, cfg.topology, cfg.seed)
    k_eff = effective_degree(G)

    # pi_ref: mean Δπ (best lax − strict) for H-firms at t=0
    _fH0, _fL0 = count_firms(firm_loc, firm_type, N)
    _ps0, _     = equilibrium_prices_with_trade(
        _fH0, _fL0, sigma, P, W, cfg.c_H, cfg.c_L, t, c_trade, cfg.tau_BA, cfg.a, cfg.b,
    )
    _piH0, _ = firm_variable_profits(_ps0, _fH0, _fL0, sigma, P, cfg.c_H, cfg.c_L, t, cfg.b)
    strict_idx = np.where(sigma == 1)[0]
    _deltas = []
    for _i in strict_idx:
        _lax_nbrs = [_j for _j in range(N) if W[_i, _j] > 0 and sigma[_j] == 0]
        if _lax_nbrs:
            _deltas.append(float(np.max(_piH0[_lax_nbrs])) - float(_piH0[_i]))
    pi_ref_val = max(float(np.mean(_deltas)), 1.0) if _deltas else 1.0

    p = Params(
        N=N, M=M, k=cfg.k, topology=cfg.topology,
        a=cfg.a, b=cfg.b, c_H=cfg.c_H, c_L=cfg.c_L, t=t,
        tau=cfg.tau, c_trade=c_trade, tau_BA=cfg.tau_BA,
        delta=cfg.delta, F=cfg.F, w_bar=cfg.w_bar,
        mu=cfg.mu, lam=cfg.lam, pi_ref=pi_ref_val,
        dt=cfg.dt, T=T_SIM, seed=cfg.seed,
        mu_P=cfg.mu_P, sigma_P=cfg.sigma_P,
        s0=float(np.mean(sigma)), h0=float(np.mean(firm_type)),
    )

    rng = np.random.default_rng(cfg.seed)

    # accumulators for validation moments
    strict_H_periods  = 0.0
    n_relocations     = 0
    ss_start          = int(T_SIM * SS_FRAC)
    import_shares     = []   # quantity-weighted, SS window only
    price_gaps        = []   # accumulated whenever both policy types coexist
    h_vals, s_vals    = [], []

    for period in range(1, T_SIM + 1):
        f_H, f_L = count_firms(firm_loc, firm_type, N)
        p_star, exp_counts = equilibrium_prices_with_trade(
            f_H, f_L, sigma, P, W, p.c_H, p.c_L, p.t, p.c_trade, p.tau_BA, p.a, p.b,
        )
        pi_H, pi_L    = firm_variable_profits(p_star, f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.b)
        pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)
        T_vec         = tariff_payoffs(sigma, W, p.tau)

        welfare_pc = np.zeros(N)
        for i in range(N):
            wage   = p.w_bar * (f_H[i] + f_L[i]) / P[i]
            cs     = (p.a - p_star[i]) ** 2 / (2 * p.b)
            tax    = (p.t * f_H[i] / P[i]) if sigma[i] == 1 else 0.0
            tariff = T_vec[i] / P[i]
            damage = -p.delta * f_H[i] / P[i]
            welfare_pc[i] = wage + cs + tax + tariff + damage

        # leakage accumulator (full run)
        strict_H_periods += float(f_H[sigma == 1].sum())

        old_loc   = firm_loc.copy()
        firm_loc  = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W)
        moved     = [(m, old_loc[m], firm_loc[m]) for m in range(M) if firm_loc[m] != old_loc[m]]
        n_relocations += len(moved)

        # emission replicator
        h_old = float(np.mean(firm_type))
        h_new = emission_replicator(h_old, pi_H_bar / p.pi_ref, pi_L_bar / p.pi_ref, p.dt)
        gap   = h_new - h_old
        if abs(gap) > 0.5 / M:
            if gap > 0:
                idx = np.where(firm_type == 0)[0]
                n   = min(int(round(gap * M)), len(idx))
                if n > 0:
                    firm_type[rng.choice(idx, n, replace=False)] = 1
            else:
                idx = np.where(firm_type == 1)[0]
                n   = min(int(round(-gap * M)), len(idx))
                if n > 0:
                    firm_type[rng.choice(idx, n, replace=False)] = 0

        # policy replicator
        a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P, p_star, W, p)
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

        # price gap: accumulate whenever both policy types coexist (full run)
        strict_mask = sigma == 1
        lax_mask    = sigma == 0
        if strict_mask.any() and lax_mask.any():
            p_s = float(np.mean(p_star[strict_mask]))
            p_l = float(np.mean(p_star[lax_mask]))
            if p_l > 0:
                price_gaps.append((p_s - p_l) / p_l * 100)

        # steady-state accumulators
        if period > ss_start:
            # quantity-weighted import penetration: Q_foreign / Q_total
            imp_num = 0.0
            imp_den = 0.0
            for ii in range(N):
                p_ii = p_star[ii]
                mc_Hd = p.c_H + p.t * sigma[ii]
                q_Hd = max(0.0, p_ii - mc_Hd) * P[ii] / p.b * f_H[ii]
                q_Ld = max(0.0, p_ii - p.c_L) * P[ii] / p.b * f_L[ii]
                imp_den += q_Hd + q_Ld
                for jj in range(N):
                    if W[ii, jj] == 0 or ii == jj:
                        continue
                    w_ij = W[ii, jj]
                    bca  = p.tau_BA if (sigma[jj] == 0 and sigma[ii] == 1) else 0.0
                    mc_He = p.c_H + p.t * sigma[jj] + c_trade + bca
                    mc_Le = p.c_L + c_trade
                    if mc_He < p_ii:
                        q = (p_ii - mc_He) * P[ii] / p.b * (f_H[jj] * w_ij)
                        imp_num += q
                        imp_den += q
                    if mc_Le < p_ii:
                        q = (p_ii - mc_Le) * P[ii] / p.b * (f_L[jj] * w_ij)
                        imp_num += q
                        imp_den += q
            if imp_den > 0:
                import_shares.append(imp_num / imp_den * 100)

            h_vals.append(float(np.mean(firm_type)))
            s_vals.append(float(np.mean(sigma)))

    leakage      = (n_relocations / strict_H_periods * 100) if strict_H_periods > 0 else 0.0
    import_pen   = float(np.mean(import_shares))   if import_shares else float("nan")
    price_gap    = float(np.mean(price_gaps))       if price_gaps    else float("nan")
    h_ss         = float(np.mean(h_vals))           if h_vals        else float("nan")
    s_ss         = float(np.mean(s_vals))           if s_vals        else float("nan")

    def _in(val, lo, hi): return lo <= val <= hi

    passes = (
        _in(leakage,   *TARGET_LEAKAGE)   and
        _in(import_pen, *TARGET_IMPORT)   and
        _in(price_gap,  *TARGET_PRICE_GAP)
    )
    return {
        "c_trade":       c_trade,
        "t":             t,
        "leakage_pct":   round(leakage,    2),
        "import_pen_pct":round(import_pen, 2),
        "price_gap_pct": round(price_gap,  2),
        "h_ss":          round(h_ss,       4),
        "s_ss":          round(s_ss,       4),
        "leakage_ok":    int(_in(leakage,    *TARGET_LEAKAGE)),
        "import_ok":     int(_in(import_pen, *TARGET_IMPORT)),
        "price_gap_ok":  int(_in(price_gap,  *TARGET_PRICE_GAP)),
        "all_pass":      int(passes),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Run sweep
# ─────────────────────────────────────────────────────────────────────────────

grid   = list(itertools.product(C_TRADE_VALUES, T_VALUES))
n_runs = len(grid)
results = []

print(f"Calibration sweep: {len(C_TRADE_VALUES)} × {len(T_VALUES)} = {n_runs} runs  "
      f"(T={T_SIM} periods each)")
print(f"{'c_trade':>8}  {'t':>6}  {'leakage%':>9}  {'import%':>8}  {'pgap%':>7}  {'pass?':>6}")
print("─" * 60)

for idx, (c_trade, t) in enumerate(grid):
    r = run_once(c_trade, t)
    results.append(r)
    status = "✓" if r["all_pass"] else " "
    print(f"  [{status}]  c_trade={c_trade:.1f}  t={t:.1f}  "
          f"leakage={r['leakage_pct']:6.1f}%  "
          f"import={r['import_pen_pct']:6.1f}%  "
          f"pgap={r['price_gap_pct']:6.1f}%")
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)
out_path = "output/sweep_calibration.csv"
fieldnames = [
    "c_trade", "t",
    "leakage_pct", "import_pen_pct", "price_gap_pct",
    "h_ss", "s_ss",
    "leakage_ok", "import_ok", "price_gap_ok", "all_pass",
]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

passing = [r for r in results if r["all_pass"]]
print(f"\n{'─'*60}")
print(f"  {len(passing)} / {n_runs} combinations pass all three targets.")
if passing:
    print(f"  Passing combinations:")
    for r in passing:
        print(f"    c_trade={r['c_trade']}  t={r['t']}  "
              f"leakage={r['leakage_pct']}%  "
              f"import={r['import_pen_pct']}%  "
              f"pgap={r['price_gap_pct']}%  "
              f"h_ss={r['h_ss']}  s_ss={r['s_ss']}")
print(f"\n  Saved: {out_path}")
