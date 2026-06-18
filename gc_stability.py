"""
gc_stability.py  —  Task 1: Green-club stability check
--------------------------------------------------------------
Question: is (h≈0, s≈1) a stable attractor for any (delta_loc, tau_BA)
in a plausible range around the calibrated baseline?

Method
------
- mu, lam pinned at calibrated baseline (mu=1.2, lam=2.0).
- 20x20 grid over (delta_loc, tau_BA), each spanning ~0.3x-3x baseline.
- Each cell initialized at (h0=0.05, s0=0.95) -- just inside the green club.
- Run to T=1000 (per convergence-horizon finding burn-in=200 is sufficient;
  we use the second-half mean as h_ss/s_ss, consistent with trajectory.py).
- Record final s. GC-stable mask: s_final > 0.8.

No dynamics are modified. Uses trajectory.run_trajectory() unchanged.
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
# Calibrated baseline (params.py module-level values)
# ─────────────────────────────────────────────────────────────────────────────

BASE_MU       = 1.2
BASE_LAM      = 2.0
BASE_DELTA_LOC = 2000.0
BASE_TAU_BA    = 5.0

T_SIM  = 1000
SEED   = 42
H0_GC  = 0.05
S0_GC  = 0.95

GRID_N = 20
DELTA_LOC_GRID = np.logspace(np.log10(BASE_DELTA_LOC * 0.3),
                              np.log10(BASE_DELTA_LOC * 3.0), GRID_N)
TAU_BA_GRID    = np.logspace(np.log10(BASE_TAU_BA * 0.3),
                              np.log10(BASE_TAU_BA * 3.0), GRID_N)

GC_THRESHOLD = 0.8   # s_final > 0.8 => GC-stable

os.makedirs("output", exist_ok=True)


def _worker(args):
    i, j, delta_loc, tau_BA = args
    h_ss, s_ss, h_final, s_final = run_trajectory(
        mu=BASE_MU, lam=BASE_LAM,
        delta_loc=delta_loc, tau_BA=tau_BA,
        h0=H0_GC, s0=S0_GC,
        T=T_SIM, seed=SEED,
    )
    return i, j, h_ss, s_ss, h_final, s_final


def main():
    jobs = [
        (i, j, dloc, tba)
        for i, dloc in enumerate(DELTA_LOC_GRID)
        for j, tba in enumerate(TAU_BA_GRID)
    ]
    total = len(jobs)
    print("=" * 72)
    print(f"TASK 1: Green-club stability  —  {GRID_N}x{GRID_N} (delta_loc, tau_BA) grid")
    print(f"        mu={BASE_MU} (pinned), lam={BASE_LAM} (pinned)")
    print(f"        init (h0={H0_GC}, s0={S0_GC}), T={T_SIM}, seed={SEED}")
    print(f"        delta_loc range: [{DELTA_LOC_GRID[0]:.0f}, {DELTA_LOC_GRID[-1]:.0f}]"
          f"  (baseline {BASE_DELTA_LOC:.0f})")
    print(f"        tau_BA range:    [{TAU_BA_GRID[0]:.2f}, {TAU_BA_GRID[-1]:.2f}]"
          f"  (baseline {BASE_TAU_BA:.2f})")
    print(f"        {total} trajectories, using {cpu_count()} cores")
    print("=" * 72)

    s_final_grid = np.full((GRID_N, GRID_N), np.nan)
    h_final_grid = np.full((GRID_N, GRID_N), np.nan)

    t0 = time.time()
    with Pool(processes=cpu_count()) as pool:
        for n_done, (i, j, h_ss, s_ss, h_final, s_final) in enumerate(
            pool.imap_unordered(_worker, jobs), start=1
        ):
            s_final_grid[i, j] = s_final
            h_final_grid[i, j] = h_final
            elapsed = time.time() - t0
            eta = (elapsed / n_done) * (total - n_done)
            print(f"\r  [{n_done:>4}/{total}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s   ",
                  end="", flush=True)
    print()

    gc_mask = s_final_grid > GC_THRESHOLD

    # ── Bounding box of GC-stable region ────────────────────────────────────
    if gc_mask.any():
        rows, cols = np.where(gc_mask)
        i_lo, i_hi = rows.min(), rows.max()
        j_lo, j_hi = cols.min(), cols.max()
        dloc_lo, dloc_hi = DELTA_LOC_GRID[i_lo], DELTA_LOC_GRID[i_hi]
        tba_lo,  tba_hi  = TAU_BA_GRID[j_lo],  TAU_BA_GRID[j_hi]
        # Representative point: interior of bounding box, nearest grid cell to centre
        i_mid = rows[np.argmin(np.abs(rows - rows.mean()) + np.abs(cols - cols.mean()))]
        # pick an actual GC-stable cell closest to the centroid
        centroid_i, centroid_j = rows.mean(), cols.mean()
        dists = (rows - centroid_i) ** 2 + (cols - centroid_j) ** 2
        best_idx = np.argmin(dists)
        rep_i, rep_j = rows[best_idx], cols[best_idx]
        rep_delta_loc = float(DELTA_LOC_GRID[rep_i])
        rep_tau_BA    = float(TAU_BA_GRID[rep_j])

        bbox = {
            "delta_loc_min": float(dloc_lo), "delta_loc_max": float(dloc_hi),
            "tau_BA_min": float(tba_lo),     "tau_BA_max": float(tba_hi),
        }
        n_stable = int(gc_mask.sum())
        print(f"\nGC-STABLE region found: {n_stable}/{total} cells "
              f"({100*n_stable/total:.1f}%)")
        print(f"  Bounding box: delta_loc in [{dloc_lo:.0f}, {dloc_hi:.0f}], "
              f"tau_BA in [{tba_lo:.2f}, {tba_hi:.2f}]")
        print(f"  Representative point: delta_loc={rep_delta_loc:.1f}, "
              f"tau_BA={rep_tau_BA:.2f}")
        gc_found = True
    else:
        bbox = None
        rep_delta_loc = None
        rep_tau_BA = None
        gc_found = False
        print("\nNO GC-STABLE region found anywhere in the grid.")
        print("(h≈0, s≈1) is NOT a stable attractor for any (delta_loc, tau_BA) "
              "in the tested range, even when initialized inside the green club.")

    # ── Plot: heatmap of final s ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    im0 = axes[0].imshow(
        s_final_grid, origin="lower", aspect="auto", cmap="viridis",
        vmin=0, vmax=1,
        extent=[0, GRID_N, 0, GRID_N],
    )
    axes[0].set_xticks(np.arange(GRID_N) + 0.5)
    axes[0].set_xticklabels([f"{v:.1f}" for v in TAU_BA_GRID], rotation=90, fontsize=6)
    axes[0].set_yticks(np.arange(GRID_N) + 0.5)
    axes[0].set_yticklabels([f"{v:.0f}" for v in DELTA_LOC_GRID], fontsize=6)
    axes[0].set_xlabel("tau_BA")
    axes[0].set_ylabel("delta_loc")
    axes[0].set_title(f"Final s(T={T_SIM})  from (h0={H0_GC}, s0={S0_GC})")
    plt.colorbar(im0, ax=axes[0], label="s_final")

    im1 = axes[1].imshow(
        gc_mask.astype(float), origin="lower", aspect="auto", cmap="RdYlGn",
        vmin=0, vmax=1,
        extent=[0, GRID_N, 0, GRID_N],
    )
    axes[1].set_xticks(np.arange(GRID_N) + 0.5)
    axes[1].set_xticklabels([f"{v:.1f}" for v in TAU_BA_GRID], rotation=90, fontsize=6)
    axes[1].set_yticks(np.arange(GRID_N) + 0.5)
    axes[1].set_yticklabels([f"{v:.0f}" for v in DELTA_LOC_GRID], fontsize=6)
    axes[1].set_xlabel("tau_BA")
    axes[1].set_ylabel("delta_loc")
    axes[1].set_title(f"GC-stable mask  (s_final > {GC_THRESHOLD})")

    fig.suptitle(
        f"Task 1: Green-club stability  —  mu={BASE_MU}, lam={BASE_LAM} (pinned baseline)\n"
        f"delta_loc, tau_BA each spanning baseline x[0.3, 3.0]",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join("output", "task1_gc_stability.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Persist results for downstream tasks / summary.json ────────────────
    result = {
        "gc_found": gc_found,
        "bbox": bbox,
        "representative_delta_loc": rep_delta_loc,
        "representative_tau_BA": rep_tau_BA,
        "n_stable_cells": int(gc_mask.sum()),
        "n_total_cells": int(total),
        "delta_loc_grid": DELTA_LOC_GRID.tolist(),
        "tau_BA_grid": TAU_BA_GRID.tolist(),
        "baseline": {
            "mu": BASE_MU, "lam": BASE_LAM,
            "delta_loc": BASE_DELTA_LOC, "tau_BA": BASE_TAU_BA,
        },
    }
    with open("output/task1_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved: output/task1_result.json")

    return result


if __name__ == "__main__":
    main()
