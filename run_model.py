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
from model.market import equilibrium_prices_with_trade, firm_variable_profits
from model.firms import count_firms, relocate_firms, emission_replicator, average_profits
from model.jurisdictions import (
    tariff_payoffs, payoff_matrix, network_correction, policy_replicator,
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

# pi_ref: mean Δπ seen by a relocating H-firm at t=0 — eq. (3.17).
# For each strict jurisdiction i, find its best lax neighbour j and take
# Δπ = pi_H[j] - pi_H[i].  This matches the local-neighbour relocation rule
# and ensures mu_ij = Δπ/pi_ref ≈ 1 for a typical first-period move.
_fH0, _fL0 = count_firms(firm_loc, firm_type, N)
_ps0, _     = equilibrium_prices_with_trade(
    _fH0, _fL0, sigma, P, W,
    cfg.c_H, cfg.c_L, cfg.t, cfg.c_trade, cfg.tau_BA, cfg.a, cfg.b,
)
_piH0, _   = firm_variable_profits(_ps0, _fH0, _fL0, sigma, P, cfg.c_H, cfg.c_L, cfg.t, cfg.b)
strict_idx = np.where(sigma == 1)[0]
_deltas = []
for _i in strict_idx:
    _lax_nbrs = [_j for _j in range(N) if W[_i, _j] > 0 and sigma[_j] == 0]
    if _lax_nbrs:
        _deltas.append(float(np.max(_piH0[_lax_nbrs])) - float(_piH0[_i]))
pi_ref_val = max(float(np.mean(_deltas)), 1.0) if _deltas else 1.0

# Params instance for model/ functions
p = Params(
    N=N, M=M, k=cfg.k, topology=cfg.topology,
    a=cfg.a, b=cfg.b, c_H=cfg.c_H, c_L=cfg.c_L, t=cfg.t,
    tau=cfg.tau, c_trade=cfg.c_trade, tau_BA=cfg.tau_BA,
    delta=cfg.delta, F=cfg.F, w_bar=cfg.w_bar,
    mu=cfg.mu, lam=cfg.lam, pi_ref=pi_ref_val,
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
_val_price_gaps   = []   # price gap whenever both policy types coexist
_val_import_share = []   # quantity-weighted import share, SS window (last 20%)
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

    # (i) Equilibrium prices — eqs. (3.12) + trade
    f_H, f_L = count_firms(firm_loc, firm_type, N)
    p_star, exp_counts = equilibrium_prices_with_trade(
        f_H, f_L, sigma, P, W,
        p.c_H, p.c_L, p.t, p.c_trade, p.tau_BA, p.a, p.b,
    )

    # (ii) Firm profits — eqs. (3.14)–(3.15)
    pi_H, pi_L = firm_variable_profits(p_star, f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.b)
    pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)

    # Tariff payoffs — eq. (3.6)
    T_vec = tariff_payoffs(sigma, W, p.tau)

    # Per-capita welfare — eq. (3.22)
    welfare_pc = np.zeros(N)
    wage_arr   = np.zeros(N)
    cs_arr     = np.zeros(N)
    tax_arr    = np.zeros(N)
    tariff_arr = np.zeros(N)
    damage_arr = np.zeros(N)
    for i in range(N):
        wage           = p.w_bar * (f_H[i] + f_L[i]) / P[i]
        cs             = (p.a - p_star[i]) ** 2 / (2 * p.b)
        tax            = (p.t * f_H[i] / P[i]) if sigma[i] == 1 else 0.0
        tariff         = T_vec[i] / P[i]
        damage         = -p.delta * f_H[i] / P[i]
        welfare_pc[i]  = wage + cs + tax + tariff + damage
        wage_arr[i]    = wage
        cs_arr[i]      = cs
        tax_arr[i]     = tax
        tariff_arr[i]  = tariff
        damage_arr[i]  = damage

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
            _p_ii = p_star[_ii]
            _mc_Hd = p.c_H + p.t * sigma[_ii]
            _imp_den += max(0.0, _p_ii - _mc_Hd) * P[_ii] / p.b * f_H[_ii]
            _imp_den += max(0.0, _p_ii - p.c_L)  * P[_ii] / p.b * f_L[_ii]
            for _jj in range(N):
                if W[_ii, _jj] == 0 or _ii == _jj:
                    continue
                _w = W[_ii, _jj]
                _bca = p.tau_BA if (sigma[_jj] == 0 and sigma[_ii] == 1) else 0.0
                _mc_He = p.c_H + p.t * sigma[_jj] + p.c_trade + _bca
                _mc_Le = p.c_L + p.c_trade
                if _mc_He < _p_ii:
                    _q = ((_p_ii - _mc_He) * P[_ii] / p.b) * (f_H[_jj] * _w)
                    _imp_num += _q
                    _imp_den += _q
                if _mc_Le < _p_ii:
                    _q = ((_p_ii - _mc_Le) * P[_ii] / p.b) * (f_L[_jj] * _w)
                    _imp_num += _q
                    _imp_den += _q
        if _imp_den > 0:
            _val_import_share.append(_imp_num / _imp_den * 100)

    # (iii) Relocation — eq. (3.17)
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
        })

    # (iv-a) Emission replicator — eq. (3.18)
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
    h_act = float(np.mean(firm_type))

    # (iv-b) Policy replicator — eqs. (3.24)–(3.33)
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P, p_star, W, p)
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
    for i in range(N):
        rows_jurisdiction.append({
            "period":        period,
            "jurisdiction":  JNAMES[i],
            "policy":        "S" if sigma[i] else "L",
            "population":    round(float(P[i]), 2),
            "f_H":           int(f_H[i]),
            "f_L":           int(f_L[i]),
            "f_total":       int(f_H[i] + f_L[i]),
            "f_imported":    round(float(exp_counts[i]), 4),
            "price":         round(float(p_star[i]), 4),
            "pi_H":          round(float(pi_H[i]), 4),
            "pi_L":          round(float(pi_L[i]), 4),
            "tariff_total":  round(float(T_vec[i]), 4),
            "wage_pc":       round(float(wage_arr[i]), 6),
            "cs_pc":         round(float(cs_arr[i]), 6),
            "tax_pc":        round(float(tax_arr[i]), 6),
            "tariff_pc":     round(float(tariff_arr[i]), 6),
            "damage_pc":     round(float(damage_arr[i]), 6),
            "welfare_pc":    round(float(welfare_pc[i]), 6),
        })

    # ── aggregate record ──────────────────────────────────────────────────────
    rows_aggregate.append({
        "period":           period,
        "h":                round(h_act, 4),
        "s":                round(s_act, 4),
        "pi_H_bar":         round(float(pi_H_bar), 4),
        "pi_L_bar":         round(float(pi_L_bar), 4),
        "pi_H_bar_norm":    round(float(pi_H_bar / p.pi_ref), 6),
        "pi_L_bar_norm":    round(float(pi_L_bar / p.pi_ref), 6),
        "mean_price":       round(float(np.mean(p_star)), 4),
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

# Use last 20% of periods as "steady state" window
_ss_start = int(cfg.T * 0.8)
_agg_ss   = _agg[_agg["period"] > _ss_start]
_jur_ss   = _jur[_jur["period"] > _ss_start]
_rel_ss   = _rel[_rel["period"] > _ss_start] if len(_rel) else _rel

# 1. Carbon leakage rate
#    Fraction of H-firms that relocated from strict → lax over the full run,
#    relative to total H-firm-periods spent in strict jurisdictions.
_strict_H_periods = _jur["f_H"][_jur["policy"] == "S"].sum()
_n_relocations    = len(_rel)
leakage_rate = (_n_relocations / _strict_H_periods * 100) if _strict_H_periods > 0 else 0.0

# 2. Import penetration (quantity-weighted, steady state)
#    Q_foreign / Q_total where Q = (p* - mc) * P/b per firm; accumulated in loop.
import_penetration = float(np.mean(_val_import_share)) if _val_import_share else float("nan")

# 3. Strict / lax price gap (all periods where both types coexist)
#    Falls back gracefully when s collapses to 0 or 1 in steady state.
price_gap_pct = float(np.mean(_val_price_gaps)) if _val_price_gaps else float("nan")

# Targets
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
    "pi_H_bar", "pi_L_bar", "pi_H_bar_norm", "pi_L_bar_norm",
    "mean_price", "mean_welfare", "n_relocations", "n_policy_changes", "b_SL",
])
write_csv("jurisdiction.csv", rows_jurisdiction, [
    "period", "jurisdiction", "policy", "population",
    "f_H", "f_L", "f_total", "f_imported", "price",
    "pi_H", "pi_L", "tariff_total",
    "wage_pc", "cs_pc", "tax_pc", "tariff_pc", "damage_pc", "welfare_pc",
])
write_csv("relocations.csv", rows_relocations, [
    "period", "firm_id", "from_jurisdiction", "to_jurisdiction",
    "pi_H_origin", "pi_H_dest", "delta_pi",
])
write_csv("policy_changes.csv", rows_policy, [
    "period", "jurisdiction", "policy_before", "policy_after", "s_before", "s_after",
])
