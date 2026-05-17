"""
Run a single simulation.  Set all parameters and initial conditions in params.py.
Saves four CSV files to output/ at the end of the run.
"""

import csv
import os
import sys
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
# Build state arrays from params.py
# ─────────────────────────────────────────────────────────────────────────────

N      = cfg.N
M      = len(cfg.firms_init)
JNAMES = [f"J{i+1}" for i in range(N)]

firm_loc  = np.array([f[0] for f in cfg.firms_init])
firm_type = np.array([1 if f[1] == "H" else 0 for f in cfg.firms_init])
sigma     = np.array(cfg.sigma_init, dtype=int)
P         = np.array(cfg.P_init, dtype=float)

G, W  = build_network(N, cfg.k, cfg.topology, cfg.seed)
k_eff = effective_degree(G)

p = Params(
    N=N, M=M, k=cfg.k, topology=cfg.topology,
    a=cfg.a, b=cfg.b, c_H=cfg.c_H, c_L=cfg.c_L, t=cfg.t,
    tau=cfg.tau, c_trade=cfg.c_trade, tau_BA=cfg.tau_BA,
    delta=cfg.delta, F=cfg.F, w_bar=cfg.w_bar, alpha=cfg.alpha,
    mu=cfg.mu, lam=cfg.lam,
    dt=cfg.dt, T=cfg.T, seed=cfg.seed,
    mu_P=cfg.mu_P, sigma_P=cfg.sigma_P,
    s0=float(np.mean(sigma)),
    h0=float(np.mean(firm_type)),
)

rng = np.random.default_rng(cfg.seed)

# CSV record lists
rows_aggregate    = []
rows_jurisdiction = []
rows_relocations  = []
rows_policy       = []

# Validation moment accumulators (populated in the loop)
_val_price_gaps   = []
_val_import_share = []
_val_ss_thresh    = int(cfg.T * 0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Progress bar helper
# ─────────────────────────────────────────────────────────────────────────────

def _progress(period, total, width=40):
    filled = int(width * period / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * period / total
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  period {period}/{total}")
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

for period in range(1, cfg.T + 1):

    # (i) Wages and goods-market equilibrium — §3.4, §3.7
    f_H, f_L = count_firms(firm_loc, firm_type, N)
    wages, p_star, q_H, q_L = solve_wages(f_H, f_L, sigma, P, W, p)

    # (ii) Fiscal revenues and per-capita welfare — eqs. (3.22)–(3.26)
    TR         = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
    welfare_pc = per_capita_welfare(f_H, sigma, P, p_star, wages, TR, p)

    # (iii) Firm profits — eqs. (3.16)–(3.17)
    pi_H, pi_L         = firm_variable_profits(q_H, q_L, P, p)
    pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)

    # ── validation moment accumulators ───────────────────────────────────────
    _smask = sigma == 1
    _lmask = sigma == 0
    if _smask.any() and _lmask.any():
        _ps_v = float(np.mean(p_star[_smask]))
        _pl_v = float(np.mean(p_star[_lmask]))
        if _pl_v > 0:
            _val_price_gaps.append((_ps_v - _pl_v) / _pl_v * 100)
    if period > _val_ss_thresh:
        _imp_num = 0.0
        _imp_den = 0.0
        for _ii in range(N):
            _imp_den += f_H[_ii] * float(q_H[_ii, _ii]) + f_L[_ii] * float(q_L[_ii, _ii])
            for _jj in range(N):
                if _jj != _ii and W[_ii, _jj] > 0:
                    _q = f_H[_jj] * float(q_H[_ii, _jj]) + f_L[_jj] * float(q_L[_ii, _jj])
                    _imp_num += _q
                    _imp_den += _q
        if _imp_den > 0:
            _val_import_share.append(_imp_num / _imp_den * 100)

    # (iv) Relocation — eq. (3.17)
    old_loc  = firm_loc.copy()
    firm_loc = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W)
    moved    = [(m, old_loc[m], firm_loc[m]) for m in range(M) if firm_loc[m] != old_loc[m]]
    for m, src, dst in moved:
        rows_relocations.append({
            "period":            period,
            "firm_id":           m,
            "from_jurisdiction": JNAMES[src],
            "to_jurisdiction":   JNAMES[dst],
            "pi_H_origin":       round(float(pi_H[src]), 4),
            "pi_H_dest":         round(float(pi_H[dst]), 4),
            "delta_pi":          round(float(pi_H[dst] - pi_H[src]), 4),
            "is_leakage":        int(sigma[src] == 1 and sigma[dst] == 0),
        })

    # (v-a) Emission replicator — eq. (3.18)
    h_old = float(np.mean(firm_type))
    h_new = emission_replicator(h_old, pi_H_bar, pi_L_bar, p.dt)
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
    h_act = float(np.mean(firm_type))

    # (v-b) Policy replicator — eqs. (3.28)–(3.34)
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P, p_star, W, wages, q_H, q_L, p)
    b_SL  = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
    s_old = float(np.mean(sigma))
    s_new = policy_replicator(s_old, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt)

    gap_s    = s_new - s_old
    n_change = int(round(abs(gap_s) * N))
    policy_records = []
    if n_change > 0:
        if gap_s > 0:
            candidates = np.where(sigma == 0)[0]
            n = min(n_change, len(candidates))
            chosen = rng.choice(candidates, n, replace=False)
            sigma[chosen] = 1
            for c in chosen:
                policy_records.append({
                    "period": period, "jurisdiction": JNAMES[c],
                    "policy_before": "L", "policy_after": "S",
                    "s_before": round(s_old, 4), "s_after": round(s_new, 4),
                })
        else:
            candidates = np.where(sigma == 1)[0]
            n = min(n_change, len(candidates))
            chosen = rng.choice(candidates, n, replace=False)
            sigma[chosen] = 0
            for c in chosen:
                policy_records.append({
                    "period": period, "jurisdiction": JNAMES[c],
                    "policy_before": "S", "policy_after": "L",
                    "s_before": round(s_old, 4), "s_after": round(s_new, 4),
                })
    rows_policy.extend(policy_records)
    s_act = float(np.mean(sigma))

    # ── per-jurisdiction records ──────────────────────────────────────────────
    P_safe = np.maximum(P, 1.0)
    for i in range(N):
        # total imports into market i
        f_imported_i = sum(
            f_H[j] * float(q_H[i, j]) + f_L[j] * float(q_L[i, j])
            for j in range(N) if j != i and W[i, j] > 0
        )
        cs_i     = (p.a - float(p_star[i])) ** 2 / (2 * p.b)
        damage_i = p.delta * float(f_H[i]) / float(P_safe[i])
        rows_jurisdiction.append({
            "period":        period,
            "jurisdiction":  JNAMES[i],
            "policy":        "S" if sigma[i] else "L",
            "population":    round(float(P[i]), 2),
            "f_H":           int(f_H[i]),
            "f_L":           int(f_L[i]),
            "f_total":       int(f_H[i] + f_L[i]),
            "f_imported":    round(f_imported_i, 4),
            "price":         round(float(p_star[i]), 4),
            "wage":          round(float(wages[i]), 6),
            "pi_H":          round(float(pi_H[i]), 4),
            "pi_L":          round(float(pi_L[i]), 4),
            "TR":            round(float(TR[i]), 4),
            "cs_pc":         round(cs_i, 6),
            "damage_pc":     round(damage_i, 6),
            "welfare_pc":    round(float(welfare_pc[i]), 6),
        })

    # ── aggregate record ──────────────────────────────────────────────────────
    rows_aggregate.append({
        "period":           period,
        "h":                round(h_act, 4),
        "s":                round(s_act, 4),
        "pi_H_bar":         round(float(pi_H_bar), 4),
        "pi_L_bar":         round(float(pi_L_bar), 4),
        "mean_price":       round(float(np.mean(p_star)), 4),
        "mean_wage":        round(float(np.mean(wages)), 6),
        "mean_welfare":     round(float(np.mean(welfare_pc)), 6),
        "n_relocations":    len(moved),
        "n_policy_changes": len(policy_records),
        "b_SL":             round(float(b_SL), 6),
    })

    _progress(period, cfg.T)

print()  # newline after progress bar

# ─────────────────────────────────────────────────────────────────────────────
# Validation report — empirical moment checks
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

_agg = pd.DataFrame(rows_aggregate)
_jur = pd.DataFrame(rows_jurisdiction)
_rel = pd.DataFrame(rows_relocations)

_ss_start = int(cfg.T * 0.8)
_agg_ss   = _agg[_agg["period"] > _ss_start]
_jur_ss   = _jur[_jur["period"] > _ss_start]
_rel_ss   = _rel[_rel["period"] > _ss_start] if len(_rel) else _rel

# 1. Carbon leakage rate
_strict_H_periods = _jur["f_H"][_jur["policy"] == "S"].sum()
_n_leakage_moves  = int(_rel["is_leakage"].sum()) if len(_rel) else 0
leakage_rate = (_n_leakage_moves / _strict_H_periods * 100) if _strict_H_periods > 0 else 0.0

# 2. Import penetration (quantity-weighted, steady state)
import_penetration = float(np.mean(_val_import_share)) if _val_import_share else float("nan")

# 3. Strict / lax price gap (all periods where both types coexist)
price_gap_pct = float(np.mean(_val_price_gaps)) if _val_price_gaps else float("nan")

_TARGET_LEAKAGE   = (5,  20)
_TARGET_IMPORT    = (20, 40)
_TARGET_PRICE_GAP = (10, 30)

def _flag(val, lo, hi):
    if lo <= val <= hi:
        return "✓  in target range"
    elif val < lo:
        return f"↓  below target  (increase {'mu' if lo==5 else 'c_trade' if lo==20 else 't'})"
    else:
        return f"↑  above target  (decrease {'mu' if lo==5 else 'c_trade' if lo==20 else 't'})"

SEP2 = "─" * 66
print(f"\n{SEP2}")
print(f"  VALIDATION REPORT")
print(SEP2)
print(f"  {'Moment':<35} {'Value':>8}   {'Target':>12}   Status")
print(f"  {'-'*35} {'-'*8}   {'-'*12}   {'-'*22}")
print(f"  {'Carbon leakage rate (%)':<35} {leakage_rate:>8.1f}   {'5 – 20 %':>12}   {_flag(leakage_rate, *_TARGET_LEAKAGE)}")
print(f"  {'Import penetration (%)':<35} {import_penetration:>8.1f}   {'20 – 40 %':>12}   {_flag(import_penetration, *_TARGET_IMPORT)}")
print(f"  {'Strict/lax price gap (%)':<35} {price_gap_pct:>8.1f}   {'10 – 30 %':>12}   {_flag(price_gap_pct, *_TARGET_PRICE_GAP)}")
print(SEP2)
print(f"  Steady-state window: periods {_ss_start+1}–{cfg.T}")
print(f"  Final h = {_agg_ss['h'].mean():.3f}   final s = {_agg_ss['s'].mean():.3f}")
print(SEP2)

# ─────────────────────────────────────────────────────────────────────────────
# Write CSVs
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)

def write_csv(filename, rows, fieldnames):
    path = os.path.join("output", filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved  {path}  ({len(rows)} rows)")

print(f"Saving CSVs to output/")
write_csv("aggregate.csv", rows_aggregate, [
    "period", "h", "s",
    "pi_H_bar", "pi_L_bar",
    "mean_price", "mean_wage", "mean_welfare",
    "n_relocations", "n_policy_changes", "b_SL",
])
write_csv("jurisdiction.csv", rows_jurisdiction, [
    "period", "jurisdiction", "policy", "population",
    "f_H", "f_L", "f_total", "f_imported", "price",
    "wage", "pi_H", "pi_L", "TR",
    "cs_pc", "damage_pc", "welfare_pc",
])
write_csv("relocations.csv", rows_relocations, [
    "period", "firm_id", "from_jurisdiction", "to_jurisdiction",
    "pi_H_origin", "pi_H_dest", "delta_pi", "is_leakage",
])
write_csv("policy_changes.csv", rows_policy, [
    "period", "jurisdiction", "policy_before", "policy_after", "s_before", "s_after",
])
