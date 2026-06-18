"""
basin_movement.py  —  Task 2: does (mu, lambda) move the GC basin boundary?
----------------------------------------------------------------------------
Only runs because Task 1 found a GC-stable region (100% of tested grid).

Method
------
- Fix (delta_loc, tau_BA) at the Task 1 representative point
  (delta_loc=1785.8, tau_BA=4.46).
- 4x4 grid over (mu, lambda): each axis at baseline x {0.1, 0.5, 2, 10}
  (baseline mu=1.2, lam=2.0) -- ~1.5 orders of magnitude span.
- Corner stability pre-check: one trajectory at each of the 4 grid corners,
  confirm h_final/s_final in [0,1] and finite. If not, narrow range and report.
- For each of the 16 (mu, lambda) cells: 30x30 basin map over (h0, s0) in
  [0,1]x[0,1], run to T=1000, classify endpoint:
    s_final > 0.8 -> GC, s_final < 0.2 -> RTB, else -> PH.
  GC basin area = fraction of the 900 (h0,s0) cells classified GC.
- Output: heatmap of GC basin area over (mu, lambda) [log-log axes],
  2x2 panel of corner basin maps, summary stats on which axis (mu, lambda,
  or diagonal ratio) moves area more.

No dynamics modified. Uses trajectory.run_trajectory() unchanged.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from trajectory import run_trajectory

# ─────────────────────────────────────────────────────────────────────────────
# Fixed point from Task 1
# ─────────────────────────────────────────────────────────────────────────────

with open("output/task1_result.json") as f:
    TASK1 = json.load(f)

assert TASK1["gc_found"], "Task 1 found no GC-stable region — Task 2 should not run."

DELTA_LOC_FIXED = TASK1["representative_delta_loc"]
TAU_BA_FIXED    = TASK1["representative_tau_BA"]

BASE_MU  = TASK1["baseline"]["mu"]
BASE_LAM = TASK1["baseline"]["lam"]

SCALE_FACTORS = [0.1, 0.5, 2.0, 10.0]
MU_GRID  = [BASE_MU * f for f in SCALE_FACTORS]
LAM_GRID = [BASE_LAM * f for f in SCALE_FACTORS]

T_SIM = 1000
SEED  = 42

BASIN_N = 30
H0_GRID = np.linspace(0.0, 1.0, BASIN_N)
S0_GRID = np.linspace(0.0, 1.0, BASIN_N)

GC_THRESHOLD  = 0.8
RTB_THRESHOLD = 0.2

os.makedirs("output", exist_ok=True)


def classify(s_final):
    if s_final > GC_THRESHOLD:
        return "GC"
    if s_final < RTB_THRESHOLD:
        return "RTB"
    return "PH"


# ── Corner stability pre-check ─────────────────────────────────────────────

def corner_check():
    corners = [
        (MU_GRID[0],  LAM_GRID[0]),
        (MU_GRID[0],  LAM_GRID[-1]),
        (MU_GRID[-1], LAM_GRID[0]),
        (MU_GRID[-1], LAM_GRID[-1]),
    ]
    print("=" * 72)
    print("Corner stability pre-check (single trajectory each, h0=0.5, s0=0.5)")
    print("=" * 72)
    ok = True
    for mu, lam in corners:
        h_ss, s_ss, h_final, s_final = run_trajectory(
            mu=mu, lam=lam, delta_loc=DELTA_LOC_FIXED, tau_BA=TAU_BA_FIXED,
            h0=0.5, s0=0.5, T=T_SIM, seed=SEED,
        )
        finite = all(np.isfinite(x) for x in [h_ss, s_ss, h_final, s_final])
        bounded = (0.0 <= h_final <= 1.0) and (0.0 <= s_final <= 1.0)
        status = "OK" if (finite and bounded) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  mu={mu:>8.4f}  lam={lam:>8.4f}  ->  h_final={h_final:.3f}  "
              f"s_final={s_final:.3f}  [{status}]")
    print()
    return ok


# ── Basin map for one (mu, lambda) cell ─────────────────────────────────────

def _worker(args):
    mu_i, lam_j, mu, lam, h0_i, s0_j, h0, s0 = args
    h_ss, s_ss, h_final, s_final = run_trajectory(
        mu=mu, lam=lam, delta_loc=DELTA_LOC_FIXED, tau_BA=TAU_BA_FIXED,
        h0=h0, s0=s0, T=T_SIM, seed=SEED,
    )
    return mu_i, lam_j, h0_i, s0_j, s_final


def main():
    if not corner_check():
        print("Corner instability detected — narrowing the (mu, lambda) range "
              "is required before proceeding. Stopping Task 2.")
        return None

    jobs = [
        (mu_i, lam_j, mu, lam, h0_i, s0_j, h0, s0)
        for mu_i, mu in enumerate(MU_GRID)
        for lam_j, lam in enumerate(LAM_GRID)
        for h0_i, h0 in enumerate(H0_GRID)
        for s0_j, s0 in enumerate(S0_GRID)
    ]
    total = len(jobs)
    print("=" * 72)
    print(f"TASK 2: GC basin movement across (mu, lambda)")
    print(f"        fixed delta_loc={DELTA_LOC_FIXED:.1f}, tau_BA={TAU_BA_FIXED:.2f}")
    print(f"        mu grid:  {[f'{v:.3f}' for v in MU_GRID]}")
    print(f"        lam grid: {[f'{v:.3f}' for v in LAM_GRID]}")
    print(f"        {BASIN_N}x{BASIN_N} (h0,s0) basin map per cell, T={T_SIM}")
    print(f"        {total} trajectories total, using {cpu_count()} cores")
    print("=" * 72)

    gc_count = np.zeros((4, 4), dtype=int)
    s_final_maps = np.full((4, 4, BASIN_N, BASIN_N), np.nan)

    t0 = time.time()
    with Pool(processes=cpu_count()) as pool:
        for n_done, (mu_i, lam_j, h0_i, s0_j, s_final) in enumerate(
            pool.imap_unordered(_worker, jobs), start=1
        ):
            s_final_maps[mu_i, lam_j, h0_i, s0_j] = s_final
            if classify(s_final) == "GC":
                gc_count[mu_i, lam_j] += 1
            if n_done % 200 == 0 or n_done == total:
                elapsed = time.time() - t0
                eta = (elapsed / n_done) * (total - n_done)
                print(f"\r  [{n_done:>6}/{total}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s   ",
                      end="", flush=True)
    print()

    gc_area = gc_count / (BASIN_N * BASIN_N)

    print()
    print("GC basin area (fraction of (h0,s0) space converging to GC):")
    _row_label = "mu \\ lam"
    print(f"  {_row_label:>10} " + "".join(f"  {lam:>8.3f}" for lam in LAM_GRID))
    for mu_i, mu in enumerate(MU_GRID):
        row = f"  {mu:>10.3f} "
        for lam_j in range(4):
            row += f"  {gc_area[mu_i, lam_j]:>8.3f}"
        print(row)

    # ── Which axis moves area more? ──────────────────────────────────────
    mu_marginal  = gc_area.mean(axis=1)   # average over lambda, as fn of mu
    lam_marginal = gc_area.mean(axis=0)   # average over mu, as fn of lambda
    diag_vals    = np.array([gc_area[i, i] for i in range(4)])  # along mu=lam scale (same scale factor)

    mu_range  = float(mu_marginal.max() - mu_marginal.min())
    lam_range = float(lam_marginal.max() - lam_marginal.min())
    diag_range = float(diag_vals.max() - diag_vals.min())

    print()
    print(f"  Range of GC area along mu (avg over lam):    {mu_range:.3f}")
    print(f"  Range of GC area along lambda (avg over mu): {lam_range:.3f}")
    print(f"  Range of GC area along diagonal (mu,lam scaled together): {diag_range:.3f}")

    if max(mu_range, lam_range, diag_range) < 0.05:
        axis_verdict = "Neither mu nor lambda meaningfully moves the GC basin boundary (all ranges < 5pp)."
    else:
        dominant = max([("mu", mu_range), ("lambda", lam_range), ("diagonal/ratio", diag_range)],
                        key=lambda x: x[1])[0]
        axis_verdict = f"GC basin area varies most along the {dominant} axis."
    print(f"  Verdict: {axis_verdict}")

    # ── Plot 1: heatmap of GC basin area over (mu, lambda) ─────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    im = ax1.imshow(gc_area, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"{v:.3f}" for v in LAM_GRID])
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f"{v:.3f}" for v in MU_GRID])
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("mu")
    ax1.set_title(f"GC basin area over (mu, lambda)\n"
                   f"fixed delta_loc={DELTA_LOC_FIXED:.0f}, tau_BA={TAU_BA_FIXED:.2f}")
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f"{gc_area[i, j]:.2f}", ha="center", va="center",
                      color="white" if gc_area[i, j] < 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax1, label="GC basin area (fraction)")
    fig1.tight_layout()
    path1 = os.path.join("output", "task2_gc_basin_area_heatmap.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {path1}")

    # ── Plot 2: 2x2 panel of corner basin maps ──────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 10))
    corner_idx = [(0, 0), (0, 3), (3, 0), (3, 3)]
    titles = [
        f"low mu ({MU_GRID[0]:.3f}), low lam ({LAM_GRID[0]:.3f})",
        f"low mu ({MU_GRID[0]:.3f}), high lam ({LAM_GRID[3]:.3f})",
        f"high mu ({MU_GRID[3]:.3f}), low lam ({LAM_GRID[0]:.3f})",
        f"high mu ({MU_GRID[3]:.3f}), high lam ({LAM_GRID[3]:.3f})",
    ]
    for ax, (mu_i, lam_j), title in zip(axes2.flat, corner_idx, titles):
        smap = s_final_maps[mu_i, lam_j]
        im = ax.imshow(smap.T, origin="lower", extent=[0, 1, 0, 1],
                        cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("h0")
        ax.set_ylabel("s0")
        ax.set_title(f"{title}\nGC area={gc_area[mu_i, lam_j]:.3f}", fontsize=10)
        plt.colorbar(im, ax=ax, label="s_final")
    fig2.suptitle("Task 2: corner basin maps (s_final over (h0,s0))", fontsize=12)
    fig2.tight_layout()
    path2 = os.path.join("output", "task2_corner_basin_maps.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    result = {
        "delta_loc_fixed": DELTA_LOC_FIXED,
        "tau_BA_fixed": TAU_BA_FIXED,
        "mu_grid": MU_GRID,
        "lam_grid": LAM_GRID,
        "gc_area": gc_area.tolist(),
        "mu_marginal": mu_marginal.tolist(),
        "lam_marginal": lam_marginal.tolist(),
        "diag_vals": diag_vals.tolist(),
        "mu_range_pp": mu_range,
        "lam_range_pp": lam_range,
        "diag_range_pp": diag_range,
        "axis_verdict": axis_verdict,
    }
    with open("output/task2_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved: output/task2_result.json")

    return result


if __name__ == "__main__":
    main()
