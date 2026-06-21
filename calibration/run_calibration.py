"""
Phase-1 calibration grid search.

Sweeps (delta_c, t_ratio, g) and finds combinations that simultaneously hit:
  leakage_pct    ∈  5 – 20 %
  import_pen_pct ∈  8 – 12 %
  price_cv_pct   ∈ 10 – 30 %

Also runs a one-at-a-time sensitivity to identify the dominant dial per moment.

Output
------
  output/calibration_grid.csv    — full grid results
  output/baseline_params.json    — best-fitting parameter dict (moment table)

Run: python calibration/run_calibration.py
"""

import sys, os, csv, json, itertools, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "calibration"))

import numpy as np
import params as cfg
from calibration import compute_moments, TARGETS, C_H

# ─────────────────────────────────────────────────────────────────────────────
# Grid specification
# ─────────────────────────────────────────────────────────────────────────────

DELTA_C_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]          # abatement gap  Δc = c_L − c_H
T_RATIO_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]    # t / Δc
G_VALUES       = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.2, 2.3, 2.4, 2.5]  # transport cost

# 7 × 8 × 8 = 448 grid points × 5 reps ≈ 2240 runs
N_REPS    = 5
BASE_SEED = 42

# Sensitivity reference point (current default params)
REF   = {"delta_c": cfg.c_L - cfg.c_H, "t_ratio": cfg.t / (cfg.c_L - cfg.c_H), "g": cfg.g}
PERTURB = 0.25   # 25% one-at-a-time perturbation


# ─────────────────────────────────────────────────────────────────────────────
# One-at-a-time sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity():
    print(f"\n{'─'*72}")
    print(f"  ONE-AT-A-TIME SENSITIVITY")
    print(f"  Reference: {REF}")
    print(f"  Perturbation: +{PERTURB*100:.0f}% of each parameter in turn")
    print(f"{'─'*72}")

    m0 = compute_moments(**REF, n_reps=N_REPS, base_seed=BASE_SEED)
    print(f"  Reference moments: "
          f"leakage={m0['leakage_mean']:.1f}%  "
          f"import={m0['import_pen_mean']:.1f}%  "
          f"priceCV={m0['price_cv_mean']:.1f}%")
    print()
    print(f"  {'param':<12}  {'Δleakage':>10}  {'Δimport%':>10}  {'ΔpriceCV%':>10}")
    print(f"  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*10}")

    changes = {}
    for param in ["delta_c", "t_ratio", "g"]:
        kw = dict(**REF)
        kw[param] = REF[param] * (1 + PERTURB)
        m1 = compute_moments(**kw, n_reps=N_REPS, base_seed=BASE_SEED)
        changes[param] = {}
        row = {}
        for key in ["leakage", "import_pen", "price_cv"]:
            v0 = m0[key + "_mean"]; v1 = m1[key + "_mean"]
            d  = (v1 - v0) if not (np.isnan(v0) or np.isnan(v1)) else float("nan")
            row[key] = d
            changes[param][key] = d
        print(f"  {param:<12}  {row['leakage']:>+10.2f}  {row['import_pen']:>+10.2f}  {row['price_cv']:>+10.2f}")

    print()
    print(f"  Dominant dial per moment  (largest absolute change at +{PERTURB*100:.0f}%):")
    for moment in ["leakage", "import_pen", "price_cv"]:
        best = max(changes, key=lambda p: abs(changes[p].get(moment, 0.0)))
        val  = changes[best][moment]
        print(f"    {moment:<15} ←  {best}   (Δ = {val:+.2f})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "delta_c", "t_ratio", "g",
    "leakage_mean", "leakage_std",
    "import_pen_mean", "import_pen_std",
    "price_cv_mean", "price_cv_std",
    "price_iqr_mean", "price_iqr_std",
    "n_active_mean", "n_valid",
    "leakage_ok", "import_ok", "price_cv_ok", "all_pass",
]


def run_grid():
    grid = list(itertools.product(DELTA_C_VALUES, T_RATIO_VALUES, G_VALUES))
    n    = len(grid)

    print(f"{'─'*72}")
    print(f"  CALIBRATION GRID  "
          f"{len(DELTA_C_VALUES)}×Δc  ×  {len(T_RATIO_VALUES)}×t/Δc  ×  {len(G_VALUES)}×g  =  {n} points,  "
          f"{N_REPS} reps each")
    tgt = TARGETS
    print(f"  Targets: leakage {tgt['leakage']}%  import {tgt['import_pen']}%  "
          f"priceCV {tgt['price_cv']}%")
    print(f"{'─'*72}")
    print(f"  {'Δc':>4} {'t/Δc':>5} {'g':>5}   {'leak%':>6} {'imp%':>6} {'cvP%':>6}  pass")
    print(f"  {'─'*4} {'─'*5} {'─'*5}   {'─'*6} {'─'*6} {'─'*6}  ────")

    results = []
    t0 = time.time()
    for idx, (dc, tr, gv) in enumerate(grid, 1):
        m = compute_moments(dc, tr, gv, n_reps=N_REPS, base_seed=BASE_SEED)
        results.append(m)
        mark = "✓" if m["all_pass"] else " "
        print(f"  {dc:>4.1f} {tr:>5.1f} {gv:>5.2f}   "
              f"{m['leakage_mean']:>6.1f} {m['import_pen_mean']:>6.1f} {m['price_cv_mean']:>6.1f}  [{mark}]")
        if idx % 50 == 0:
            eta = (time.time() - t0) / idx * (n - idx)
            print(f"  ... {idx}/{n}  ({time.time()-t0:.0f}s elapsed, ~{eta:.0f}s remaining)")
        sys.stdout.flush()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save results + final table
# ─────────────────────────────────────────────────────────────────────────────

def save_and_report(results):
    os.makedirs("output", exist_ok=True)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_path = "output/calibration_grid.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved: {csv_path}")

    # ── Choose best candidate ──────────────────────────────────────────────
    passing = [r for r in results if r["all_pass"]]
    print(f"  {len(passing)} / {len(results)} combinations pass all three targets.")

    def _score(r):
        # Normalised distance from centre of each target band
        tgt = TARGETS
        def d(v, lo, hi): return abs(v - (lo + hi) / 2) / ((hi - lo) / 2)
        return (d(r["leakage_mean"],   *tgt["leakage"])  +
                d(r["import_pen_mean"],*tgt["import_pen"]) +
                d(r["price_cv_mean"],  *tgt["price_cv"]))

    if passing:
        best = min(passing, key=_score)
    else:
        # Fallback: maximise number of passing targets, then minimise gap
        def n_pass(r): return r["leakage_ok"] + r["import_ok"] + r["price_cv_ok"]
        max_p = max(n_pass(r) for r in results)
        cands = [r for r in results if n_pass(r) == max_p]

        def _gap(r):
            def g_val(v, lo, hi): return max(0.0, lo - v, v - hi)
            tgt = TARGETS
            return (g_val(r["leakage_mean"],    *tgt["leakage"]) +
                    g_val(r["import_pen_mean"],  *tgt["import_pen"]) +
                    g_val(r["price_cv_mean"],    *tgt["price_cv"]))
        best = min(cands, key=_gap)

    # ── Build baseline_params.json ─────────────────────────────────────────
    dc  = best["delta_c"]
    c_L = C_H + dc
    t   = best["t_ratio"] * dc
    g   = best["g"]

    baseline = {
        "a":         cfg.a,
        "b":         cfg.b,
        "c_H":       C_H,
        "c_L":       round(c_L, 4),
        "delta_c":   dc,
        "t":         round(t, 4),
        "t_over_dc": best["t_ratio"],
        "g":         g,
        "tau":       cfg.tau,
        "tau_BA":    cfg.tau_BA,
        "F":         cfg.F,
        "mu_P":      cfg.mu_P,
        "sigma_P":   cfg.sigma_P,
        "N":         cfg.N,
        "M":         cfg.M,
        "k":         cfg.k,
        "topology":  cfg.topology,
        "h0":        cfg.h0,
        "_moment_table": {
            "leakage_pct":    {
                "target": list(TARGETS["leakage"]),
                "achieved": round(best["leakage_mean"],    2),
                "std":      round(best["leakage_std"],     2),
                "in_band":  bool(best["leakage_ok"]),
                "dialed_by": "mu (relocation rate), t, delta_c",
            },
            "import_pen_pct": {
                "target": list(TARGETS["import_pen"]),
                "achieved": round(best["import_pen_mean"], 2),
                "std":      round(best["import_pen_std"],  2),
                "in_band":  bool(best["import_ok"]),
                "dialed_by": "g (transport cost)",
            },
            "price_cv_pct":   {
                "target": list(TARGETS["price_cv"]),
                "achieved": round(best["price_cv_mean"],   2),
                "std":      round(best["price_cv_std"],    2),
                "in_band":  bool(best["price_cv_ok"]),
                "dialed_by": "sigma_P (population spread), policy heterogeneity",
            },
        },
    }

    json_path = "output/baseline_params.json"
    with open(json_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Print final moment table ───────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  PROPOSED BASELINE PARAMETER DICT")
    print(f"{'─'*72}")
    print(f"  a={cfg.a}   b={cfg.b}   c_H={C_H}   c_L={c_L:.2f}   Δc={dc:.2f}")
    print(f"  t={t:.2f}   t/Δc={best['t_ratio']:.1f}   g={g:.2f}")
    print(f"  tau={cfg.tau}   tau_BA={cfg.tau_BA}   F={cfg.F}   mu_P={cfg.mu_P}   sigma_P={cfg.sigma_P}")
    print()
    print(f"  {'Moment':<22}  {'Dialed by':<25}  {'Target':>10}  {'Achieved':>9}  {'±σ':>6}  {'In band?':>8}")
    print(f"  {'─'*22}  {'─'*25}  {'─'*10}  {'─'*9}  {'─'*6}  {'─'*8}")
    rows = [
        ("leakage_pct",    "mu, t, delta_c",          "5 – 20 %",  best["leakage_mean"],    best["leakage_std"],    best["leakage_ok"]),
        ("import_pen_pct", "g (transport cost)",       "8 – 12 %",  best["import_pen_mean"], best["import_pen_std"], best["import_ok"]),
        ("price_cv_pct",   "sigma_P, policy mix",      "10 – 30 %", best["price_cv_mean"],   best["price_cv_std"],   best["price_cv_ok"]),
    ]
    for name, dial, tgt_str, val, std, ok in rows:
        mark = "✓" if ok else "✗"
        print(f"  {name:<22}  {dial:<25}  {tgt_str:>10}  {val:>9.1f}  {std:>6.2f}  {mark:>8}")
    print(f"{'─'*72}")
    print()
    print(f"  Review output/baseline_params.json and confirm before updating params.py.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_structural_check():
    """
    Inline replication of check_structural.py logic for the CURRENT params.py values.
    Prints pass/fail for all 8 conditions and aborts if any critical one fails.
    """
    import math

    p = cfg

    n_total = p.M / p.N
    n_H     = round(n_total * p.h0)
    n_L     = round(n_total * (1 - p.h0))
    P_med   = math.exp(p.mu_P)
    P_b     = P_med / p.b

    def cournot(groups, a):
        active = list(range(len(groups)))
        costs  = [c for c, _ in groups]
        counts = [n for _, n in groups]
        while True:
            n_tot  = sum(counts[k] for k in active)
            mc_sum = sum(costs[k] * counts[k] for k in active)
            p_eq   = (a + mc_sum) / (n_tot + 1)
            new    = [k for k in active if costs[k] < p_eq]
            if len(new) == len(active):
                return p_eq, active
            if not new:
                return a, []
            active = new

    def var_profit(p_star, cost):
        q = P_b * max(p_star - cost, 0.0)
        return P_b * max(p_star - cost, 0.0) ** 2

    def markup(p_star, cost):
        return (p_star - cost) / p_star if p_star > 0 else float("nan")

    p_lax, act_lax = cournot([(p.c_H, n_H), (p.c_L, n_L)], p.a)
    H_active_lax   = 0 in act_lax
    L_active_lax   = 1 in act_lax
    pi_H_lax       = var_profit(p_lax, p.c_H)

    c_H_strict = p.c_H + p.t
    p_strict, act_strict = cournot([(c_H_strict, n_H), (p.c_L, n_L)], p.a)
    H_active_str = 0 in act_strict
    L_active_str = 1 in act_strict
    pi_L_str     = var_profit(p_strict, p.c_L)
    p_L_only, _  = cournot([(p.c_L, n_L)], p.a)

    export_margin_L = p_lax - p.c_L

    cs_lax    = (p.a - p_lax)    ** 2 / (2 * p.b)
    fH_density = n_H / P_med
    local_dmg  = p.delta_loc * fH_density

    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  STRUCTURAL CHECK  (params.py current values)")
    print(SEP)
    print(f"  a={p.a}  b={p.b}  c_H={p.c_H}  c_L={p.c_L}  t={p.t}  g={p.g}  F={p.F}")
    print(f"  n_H={n_H}  n_L={n_L}  P_med={P_med:.0f}")
    print()

    checks = [
        ("C1  Both H and L active in lax autarky",
         H_active_lax and L_active_lax,
         f"p*_lax={p_lax:.3f}  H={'✓' if H_active_lax else '✗'}  L={'✓' if L_active_lax else '✗'}",
         True),
        # Non-critical: FULL crowd-out (c_H+t >= p*_L-only) makes strict so dominant
        # it suppresses the race-to-bottom regime. Partial crowd-out (H weakly present
        # in strict) is what the trade moments calibrate to and what keeps the
        # phi/mu-lambda bifurcation alive. Informational only.
        ("C2  H crowded out in strict autarky (full crowd-out; partial is fine)",
         c_H_strict >= p_L_only,
         f"c_H+t={c_H_strict:.2f}  p*_L-only={p_L_only:.3f}",
         False),
        ("C3  L profitable in strict",
         pi_L_str > p.F,
         f"var={pi_L_str:.1f}  F={p.F}",
         True),
        ("C4  H profitable in lax",
         pi_H_lax > p.F,
         f"var={pi_H_lax:.1f}  F={p.F}",
         True),
        ("C5  F economically meaningful (F/pi_H in 0.05–0.50)",
         0.05 <= p.F / pi_H_lax <= 0.50 if pi_H_lax > 0 else False,
         f"ratio={p.F/pi_H_lax:.4f}  (F=0 → skip)",
         False),   # non-critical with F=0
        ("C6  Carbon markup (c_H+t)/c_L in [1.0, 2.5]",
         1.0 <= c_H_strict / p.c_L <= 2.5,
         f"ratio={c_H_strict/p.c_L:.2f}",
         True),
        ("C7  Trade not blocked by g  (g < p*_lax − c_L)",
         p.g < export_margin_L,
         f"g={p.g}  margin={export_margin_L:.3f}",
         False),   # non-critical — grid sweeps g
        ("C8  Lax price markup plausible (5–35%)",
         0.05 <= markup(p_lax, p.c_L) <= 0.35,
         f"markup={markup(p_lax, p.c_L)*100:.1f}%",
         True),
    ]

    critical_failures = []
    for label, ok, detail, critical in checks:
        icon = "✓" if ok else ("✗ CRITICAL" if critical else "✗ warn")
        print(f"  [{icon:<10}]  {label}  ({detail})")
        if not ok and critical:
            critical_failures.append(label)

    print()
    print(f"  damage/CS ratio  = {local_dmg/cs_lax:.3f}  (want 0.10–0.50)")
    print(f"  t / Δc           = {p.t/(p.c_L-p.c_H):.2f}")
    print(f"  g / export margin= {p.g/export_margin_L:.2f}  (current g likely too large — grid will fix)")
    print(SEP)

    if critical_failures:
        print(f"\n  ABORT: {len(critical_failures)} critical condition(s) failed.")
        for f in critical_failures:
            print(f"    • {f}")
        print(f"  Fix params.py before running the calibration grid.\n")
        sys.exit(1)
    else:
        print(f"\n  All critical conditions pass. Proceeding to calibration.\n")


if __name__ == "__main__":
    run_structural_check()
    run_sensitivity()
    results = run_grid()
    save_and_report(results)
