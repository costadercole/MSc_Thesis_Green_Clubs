"""
Regime calibration — host benefit (phi) × local damage (delta_loc).

Goal
----
Find (phi, delta_loc) such that, at the INTERIOR of the phase diagram
(s ≈ 0.5, h ≈ 0.5), strict jurisdictions win roughly 60–70% of actual
pairwise neighbour welfare comparisons.

  win_rate = fraction of (strict_i, lax_j) neighbour pairs where W_i > W_j
           + fraction of (lax_i, strict_j) neighbour pairs where W_j > W_i
           (both count "strict wins")

Why 60–70%?
  - Below 50%: lax always wins → s collapses to 0 regardless of mu/lambda
  - Above 80%: strict always wins → s goes to 1 regardless of mu/lambda
  - 60–70%: strict has a genuine edge but firm mobility (mu) can overcome it,
    producing the bifurcation the thesis needs.  This band is the CONTESTED
    structural region that feeds experiments/phase_diagram.py (speed preset).

Why sweep phi (not delta_glob)?
  The strict-vs-lax welfare gap is set by the per-unit payoff of hosting dirty
  output, ≈ (delta_loc − phi)·E_H + fiscal:
    phi  = host benefit (jobs + non-carbon tax base) retained per unit of local
           output — the reason a lax jurisdiction WANTS to keep dirty industry.
    delta_loc = local environmental damage per unit of that output.
  So phi and delta_loc are the two opposing knobs on the win-rate.  delta_glob
  only shifts the absolute level (it hits every jurisdiction equally), so it is
  held fixed at delta_loc/4 — the convention used throughout the model.

Output
------
  output/damage_calibration.csv   — win_rate for each (phi, delta_loc) pair
  output/damage_calibration.png   — heatmap, target band contour overlaid

Run: python calibration/calibrate_damage.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from params import BASELINE
from model.network import build_network
from model.firms import init_firms, count_firms, relocate_firms, firm_type_update
from model.jurisdictions import (
    init_jurisdictions, init_populations,
    fiscal_revenues, per_capita_welfare,
)
from model.market import solve_market, firm_variable_profits


# ── Grid ──────────────────────────────────────────────────────────────────────
# Sweep phi (host benefit, x-axis) against delta_loc (local damage, y-axis).
# delta_glob is tied to delta_loc/4 per cell (model convention; level shift only).

PHI_VALUES       = list(range(0, 1301, 100))   # 0,100,…,1300 — dense: the contested band is narrow
DELTA_LOC_VALUES = [100, 200, 300, 500, 700, 1000]

N_REPS   = 5      # independent seeds per grid point
T_SAMPLE = 100    # steps to run per rep (we only need the interior, not convergence)
T_WARMUP = 30     # steps before we start recording win-rates (let firms sort a bit)

TARGET_LO, TARGET_HI = 0.60, 0.70   # desired strict win-rate band


# ── Core diagnostic ───────────────────────────────────────────────────────────

def measure_win_rate(phi: float, delta_loc: float, seed: int) -> float:
    """
    Run a short simulation at s0=0.5, h0=0.5 (interior) and return the
    fraction of neighbour pairs where the strict jurisdiction has higher welfare.

    phi       : host benefit per unit of local output (jobs + non-carbon tax base)
    delta_loc : local environmental damage per unit of output
    delta_glob is tied to delta_loc/4 (model convention; level shift only).
    """
    p = replace(
        BASELINE,
        phi=phi,
        delta_loc=delta_loc,
        delta_glob=delta_loc / 4.0,
        s0=0.5, h0=0.5,
        T=T_SAMPLE + T_WARMUP,
        seed=seed,
    )

    rng = np.random.default_rng(p.seed)
    G, W = build_network(p.N, p.k, p.topology, p.seed)
    P    = init_populations(p, rng)

    # Force s≈0.5 and h≈0.5 exactly
    sigma     = np.zeros(p.N, dtype=int)
    sigma[:p.N // 2] = 1
    rng.shuffle(sigma)

    firm_loc, firm_type = init_firms(p, rng)

    strict_wins = 0
    total_pairs = 0

    for step in range(T_WARMUP + T_SAMPLE):
        f_H, f_L         = count_firms(firm_loc, firm_type, p.N)
        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, p)
        pi_H, pi_L       = firm_variable_profits(q_H, q_L, P, p)
        TR               = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
        welfare          = per_capita_welfare(f_H, sigma, P, p_star, TR, p, q_H, f_L, q_L)

        profit = np.where(firm_type == 1, pi_H[firm_loc], pi_L[firm_loc])
        firm_loc  = relocate_firms(firm_loc, firm_type, sigma, pi_H, p, rng, W=W)
        firm_type = firm_type_update(firm_type, profit,
                                     nu=p.nu, kappa_f=p.kappa_f,
                                     dt=p.dt, rng=rng, eps=p.eps)

        if step < T_WARMUP:
            continue

        # Count pairwise wins across all neighbour edges
        for i in range(p.N):
            for j in range(p.N):
                if W[i, j] == 0 or j <= i:
                    continue
                if sigma[i] != sigma[j]:   # mixed-policy pair only
                    s_idx = i if sigma[i] == 1 else j
                    l_idx = j if sigma[i] == 1 else i
                    total_pairs += 1
                    if welfare[s_idx] > welfare[l_idx]:
                        strict_wins += 1

    return strict_wins / total_pairs if total_pairs > 0 else float("nan")


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    results = []
    n_total = len(PHI_VALUES) * len(DELTA_LOC_VALUES)

    print(f"Grid: {len(PHI_VALUES)} phi × {len(DELTA_LOC_VALUES)} delta_loc "
          f"= {n_total} points × {N_REPS} reps")
    print(f"Target win-rate: {TARGET_LO:.0%} – {TARGET_HI:.0%}")
    print()
    print(f"  {'phi':>8}  {'delta_loc':>10}  {'win_rate':>9}  {'in_band':>7}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*7}")

    for dloc in DELTA_LOC_VALUES:
        for phi in PHI_VALUES:
            rates = [measure_win_rate(phi, dloc, seed=42 + r) for r in range(N_REPS)]
            wr = float(np.mean(rates))
            wr_std = float(np.std(rates))
            in_band = TARGET_LO <= wr <= TARGET_HI
            mark = "✓" if in_band else " "
            print(f"  {phi:>8}  {dloc:>10}  {wr:>8.1%}  [{mark}]  ±{wr_std:.2%}")
            results.append(dict(phi=phi, delta_loc=dloc,
                                win_rate=round(wr, 4), win_rate_std=round(wr_std, 4),
                                in_band=in_band))

    return results


# ── Save + plot ───────────────────────────────────────────────────────────────

def save_and_plot(results):
    os.makedirs("output", exist_ok=True)

    csv_path = "output/damage_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phi", "delta_loc",
                                                "win_rate", "win_rate_std", "in_band"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # Build matrix for heatmap: rows = delta_loc (y), cols = phi (x)
    phi_vals  = sorted(set(r["phi"]       for r in results))
    dloc_vals = sorted(set(r["delta_loc"] for r in results))
    mat = np.full((len(dloc_vals), len(phi_vals)), np.nan)
    for r in results:
        i = dloc_vals.index(r["delta_loc"])
        j = phi_vals.index(r["phi"])
        mat[i, j] = r["win_rate"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, origin="lower", aspect="auto",
                   vmin=0.4, vmax=0.9, cmap="RdYlGn",
                   extent=[-0.5, len(phi_vals) - 0.5,
                           -0.5, len(dloc_vals) - 0.5])
    plt.colorbar(im, ax=ax, label="Strict win-rate (fraction of neighbour pairs)")

    # Overlay target band contour
    ax.contour(mat, levels=[TARGET_LO, TARGET_HI], colors=["blue"],
               linestyles=["--", "--"], linewidths=1.5)

    ax.set_xticks(range(len(phi_vals)))
    ax.set_xticklabels([str(v) for v in phi_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(dloc_vals)))
    ax.set_yticklabels([str(v) for v in dloc_vals])
    ax.set_xlabel("phi  (host benefit per unit output)", fontsize=11)
    ax.set_ylabel("delta_loc  (local damage per unit output)", fontsize=11)
    ax.set_title("Strict jurisdiction win-rate at interior (s=0.5, h=0.5)\n"
                 "Blue dashed lines = contested band 60–70% (where mu/lambda tips the regime)",
                 fontsize=11)
    fig.tight_layout()

    png_path = "output/damage_calibration.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    # Print passing candidates
    passing = [r for r in results if r["in_band"]]
    print(f"\n{len(passing)} / {len(results)} combinations in target band:")
    for r in passing:
        print(f"  phi={r['phi']:6}  delta_loc={r['delta_loc']:6}  "
              f"win_rate={r['win_rate']:.1%} ± {r['win_rate_std']:.1%}")
    if not passing:
        closest = min(results, key=lambda r: min(abs(r["win_rate"] - TARGET_LO),
                                                  abs(r["win_rate"] - TARGET_HI)))
        print(f"  No exact match. Closest: phi={closest['phi']}, "
              f"delta_loc={closest['delta_loc']}, win_rate={closest['win_rate']:.1%}")


if __name__ == "__main__":
    results = run_sweep()
    save_and_plot(results)
