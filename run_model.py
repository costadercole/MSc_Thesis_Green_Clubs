"""
Run a single simulation.  Set all parameters and initial conditions in params.py.
Saves four CSV files to output/ at the end of the run.
"""

import csv
import os
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
VERBOSE = N <= 10   # full per-jurisdiction printing only for small N

firm_loc  = np.array([f[0] for f in cfg.firms_init])
firm_type = np.array([1 if f[1] == "H" else 0 for f in cfg.firms_init])
sigma     = np.array(cfg.sigma_init, dtype=int)
P         = np.array(cfg.P_init, dtype=float)

G, W = build_network(N, cfg.k, cfg.topology, cfg.seed)
k_eff = effective_degree(G)

# pi_ref: variable profit of H-firm in lax jurisdictions at t=0 — eq. (3.17)
_fH0, _fL0 = count_firms(firm_loc, firm_type, N)
_ps0, _     = equilibrium_prices_with_trade(
    _fH0, _fL0, sigma, P, W,
    cfg.c_H, cfg.c_L, cfg.t, cfg.c_trade, cfg.tau_BA, cfg.a, cfg.b,
)
_piH0, _    = firm_variable_profits(_ps0, _fH0, _fL0, sigma, P, cfg.c_H, cfg.c_L, cfg.t, cfg.b)
lax_idx     = np.where(sigma == 0)[0]
pi_ref_val  = float(np.mean(_piH0[lax_idx])) if len(lax_idx) else 1.0

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

print(f"N={N}  M={M}  k={k_eff:.1f}  topology={cfg.topology}")
print(f"pi_ref = {pi_ref_val:.2f}  (H-firm variable profit in lax jurisdictions at t=0)")
print(f"s0 = {float(np.mean(sigma)):.2f}   h0 = {float(np.mean(firm_type)):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def state_summary(sigma, f_H, f_L, firm_type):
    return (f"h={np.mean(firm_type):.3f}  s={np.mean(sigma):.3f}  "
            f"f_H_total={int(f_H.sum())}  f_L_total={int(f_L.sum())}")

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 66

for period in range(1, cfg.T + 1):
    print(f"\n{SEP}")
    print(f"  PERIOD {period}")
    print(SEP)

    # (i) Equilibrium prices — eqs. (3.12) + trade
    f_H, f_L = count_firms(firm_loc, firm_type, N)
    p_star, exp_counts = equilibrium_prices_with_trade(
        f_H, f_L, sigma, P, W,
        p.c_H, p.c_L, p.t, p.c_trade, p.tau_BA, p.a, p.b,
    )

    print(f"\nStart: {state_summary(sigma, f_H, f_L, firm_type)}")

    if VERBOSE:
        print(f"\n[eq. 3.12+trade]  Prices:")
        for i in range(N):
            f_i   = int(f_H[i] + f_L[i])
            c_bar = (f_H[i] * (p.c_H + p.t * sigma[i]) + f_L[i] * p.c_L) / max(f_i, 1)
            print(f"  {JNAMES[i]} ({'S' if sigma[i] else 'L'}, "
                  f"f_dom={f_i}, f_imp≈{exp_counts[i]:.1f}):  "
                  f"c̄={c_bar:.2f}  p*={p_star[i]:.2f}")
    else:
        strict_mask = sigma == 1
        lax_mask    = sigma == 0
        print(f"  mean p* strict={np.mean(p_star[strict_mask]):.2f}"
              f"  lax={np.mean(p_star[lax_mask]):.2f}"
              f"  mean f_imp={np.mean(exp_counts):.2f}")

    # (ii) Firm profits — eqs. (3.14)–(3.15)
    pi_H, pi_L = firm_variable_profits(p_star, f_H, f_L, sigma, P, p.c_H, p.c_L, p.t, p.b)
    pi_H_bar, pi_L_bar = average_profits(firm_loc, firm_type, pi_H, pi_L, p.F)

    if VERBOSE:
        print(f"\n[eq. 3.15]  Variable profit per jurisdiction:")
        for i in range(N):
            print(f"  {JNAMES[i]}:  π_H={pi_H[i]:10.2f}   π_L={pi_L[i]:10.2f}")
    print(f"[eq. 3.14]  π̄_H={pi_H_bar:.2f}   π̄_L={pi_L_bar:.2f}"
          f"   diff={pi_H_bar - pi_L_bar:.2f}")

    # Tariff payoffs — eq. (3.6)
    T_vec = tariff_payoffs(sigma, W, p.tau)
    if VERBOSE:
        print(f"\n[eq. 3.6]  Tariff payoffs:")
        for i in range(N):
            print(f"  {JNAMES[i]}:  T={T_vec[i]:+.2f}")

    # Per-capita welfare — eq. (3.22)
    welfare_pc = np.zeros(N)
    wage_arr   = np.zeros(N)
    cs_arr     = np.zeros(N)
    tax_arr    = np.zeros(N)
    tariff_arr = np.zeros(N)
    damage_arr = np.zeros(N)

    for i in range(N):
        wage        = p.w_bar * (f_H[i] + f_L[i]) / P[i]
        cs          = (p.a - p_star[i]) ** 2 / (2 * p.b)
        tax         = (p.t * f_H[i] / P[i]) if sigma[i] == 1 else 0.0
        tariff      = T_vec[i] / P[i]
        damage      = -p.delta * f_H[i] / P[i]
        wpc         = wage + cs + tax + tariff + damage
        welfare_pc[i]  = wpc
        wage_arr[i]    = wage
        cs_arr[i]      = cs
        tax_arr[i]     = tax
        tariff_arr[i]  = tariff
        damage_arr[i]  = damage

    if VERBOSE:
        print(f"\n[eq. 3.22]  Per-capita welfare:")
        for i in range(N):
            print(f"  {JNAMES[i]} ({'S' if sigma[i] else 'L'}):  "
                  f"wage={wage_arr[i]:.3f}  CS={cs_arr[i]:.3f}  "
                  f"tax={tax_arr[i]:.3f}  tariff={tariff_arr[i]:+.3f}  "
                  f"dmg={damage_arr[i]:.3f}   W/P={welfare_pc[i]:.3f}")
    else:
        strict_mask = sigma == 1
        lax_mask    = sigma == 0
        wS = np.mean(welfare_pc[strict_mask]) if strict_mask.any() else float("nan")
        wL = np.mean(welfare_pc[lax_mask])    if lax_mask.any()   else float("nan")
        print(f"[eq. 3.22]  mean W/P:  strict={wS:.3f}   lax={wL:.3f}")

    # (iii) Relocation — eq. (3.17)
    old_loc  = firm_loc.copy()
    firm_loc = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng)
    moved    = [(m, old_loc[m], firm_loc[m]) for m in range(M) if firm_loc[m] != old_loc[m]]

    print(f"\n[eq. 3.17]  Relocations: {len(moved)}")
    if VERBOSE and moved:
        for m, src, dst in moved:
            print(f"  firm {m} [H]  {JNAMES[src]}→{JNAMES[dst]}"
                  f"   Δπ={pi_H[dst]-pi_H[src]:+.2f}")
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
    # Profits are normalised by pi_ref so the Euler step is O(1) regardless of
    # the absolute scale of profits (which depends on population size).
    h_old  = float(np.mean(firm_type))
    h_new  = emission_replicator(
        h_old,
        pi_H_bar / p.pi_ref,
        pi_L_bar / p.pi_ref,
        p.dt,
    )
    gap = h_new - h_old
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
    arrow = "↑" if h_act > h_old else ("↓" if h_act < h_old else "→")
    print(f"\n[eq. 3.18]  h: {h_old:.3f} {arrow} {h_act:.3f}"
          f"   (π̄_H/π_ref={pi_H_bar/p.pi_ref:.3f}"
          f"  π̄_L/π_ref={pi_L_bar/p.pi_ref:.3f})")

    # (iv-b) Policy replicator — eqs. (3.24)–(3.33)
    a_SS, a_SL, a_LS, a_LL = payoff_matrix(f_H, f_L, sigma, P, p_star, W, p)
    b_SL   = network_correction(a_SS, a_SL, a_LS, a_LL, k_eff)
    s_old  = float(np.mean(sigma))
    s_new  = policy_replicator(s_old, a_SS, a_SL, a_LS, a_LL, b_SL, p.dt)

    gap_s     = s_new - s_old
    n_change  = int(round(abs(gap_s) * N))
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
    arrow_s = "↑" if s_act > s_old else ("↓" if s_act < s_old else "→")
    print(f"[eq. 3.31]  s: {s_old:.3f} {arrow_s} {s_act:.3f}"
          f"   b_SL={b_SL:.4f}   changes={len(policy_records)}")

    f_H_end, f_L_end = count_firms(firm_loc, firm_type, N)
    print(f"\nEnd:   {state_summary(sigma, f_H_end, f_L_end, firm_type)}")

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

print(f"\n{SEP}")
print(f"  DONE  ({cfg.T} periods)")
print(SEP)

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

print(f"\nSaving CSVs to output/")
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
