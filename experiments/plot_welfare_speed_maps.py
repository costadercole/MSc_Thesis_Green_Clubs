"""
Plot the welfare speed maps figure (welfare_speed_maps.png) from the
already-computed exp1_speed_map.csv.

Two panels over the mu x lambda plane (log axes):
  Left  — total social welfare W_tot, coloured green-to-red, with the
           R = 0 regime boundary overlaid in black.
  Right — the coordination-failure wedge: cells where the dynamics selected
           the welfare-inferior regime (race-to-bottom) despite fundamentals
           being identical everywhere (so green club is optimal in every cell).

Run from the project root:
    python experiments/plot_welfare_speed_maps.py
"""

import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, "output", "exp1_speed_map.csv")
OUT_PATH = os.path.join(_ROOT, "output", "welfare_speed_maps.png")

# ── load CSV ────────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        rows.append({k: float(v) if k not in ("regime",) else v
                     for k, v in row.items()})

mu_vals  = sorted(set(r["mu"]  for r in rows))
lam_vals = sorted(set(r["lam"] for r in rows))
N_mu, N_lam = len(mu_vals), len(lam_vals)

mu_idx  = {v: i for i, v in enumerate(mu_vals)}
lam_idx = {v: i for i, v in enumerate(lam_vals)}

# matrices: rows = lambda (y-axis), cols = mu (x-axis)
W_mat    = np.full((N_lam, N_mu), np.nan)
R_mat    = np.full((N_lam, N_mu), np.nan)
wedge    = np.zeros((N_lam, N_mu))    # 1 where dynamics selected RTB

for r in rows:
    i = lam_idx[r["lam"]]
    j = mu_idx[r["mu"]]
    W_mat[i, j] = r["all_W_tot"] / 1e6      # in millions
    R_mat[i, j] = r["s_mean"] - r["h_mean"] # order parameter R = s - h
    if r["regime"] == "race_to_bottom":
        wedge[i, j] = 1.0

mu_arr  = np.array(mu_vals)
lam_arr = np.array(lam_vals)

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# shared log-scale tick labels
MU_TICKS  = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 20]
LAM_TICKS = [0.1, 0.2, 0.5, 1, 2, 5]

def log_extent(xv, yv):
    """pcolormesh-style edges in log space."""
    lx = np.log10(xv); ly = np.log10(yv)
    dxh = (lx[1]-lx[0])/2; dyh = (ly[1]-ly[0])/2
    xe = np.concatenate([[lx[0]-dxh], lx[:-1]+np.diff(lx)/2, [lx[-1]+dxh]])
    ye = np.concatenate([[ly[0]-dyh], ly[:-1]+np.diff(ly)/2, [ly[-1]+dyh]])
    return 10**xe, 10**ye

mu_edges, lam_edges = log_extent(mu_arr, lam_arr)


# ── Left panel: W_tot heatmap ────────────────────────────────────────────────
ax = axes[0]
vmin, vmax = np.nanmin(W_mat), np.nanmax(W_mat)
im = ax.pcolormesh(mu_edges, lam_edges, W_mat,
                   cmap="RdYlGn", vmin=vmin, vmax=vmax,
                   shading="flat")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("$W_{\\mathrm{tot}}$ (M)", fontsize=10)

# R = 0 contour
R_smooth = gaussian_filter(R_mat, sigma=0.8)
ax.contour(mu_arr, lam_arr, R_smooth, levels=[0],
           colors="black", linewidths=1.5, linestyles="-")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(MU_TICKS);  ax.set_xticklabels(MU_TICKS)
ax.set_yticks(LAM_TICKS); ax.set_yticklabels(LAM_TICKS)
ax.set_xlabel("firm mobility  $\\mu$", fontsize=11)
ax.set_ylabel("institutional adaptation  $\\lambda$", fontsize=11)
ax.set_title("Total social welfare $W_{\\mathrm{tot}}$\n(black contour: $R = 0$ boundary)",
             fontsize=11)


# ── Right panel: coordination-failure wedge ──────────────────────────────────
ax = axes[1]
# green = dynamics selected green club (optimal), red = selected RTB (suboptimal)
cmap2 = matplotlib.colors.ListedColormap(["#1a9850", "#d73027"])   # green / red
ax.pcolormesh(mu_edges, lam_edges, wedge,
              cmap=cmap2, vmin=0, vmax=1, shading="flat")

# same R = 0 boundary
ax.contour(mu_arr, lam_arr, R_smooth, levels=[0],
           colors="black", linewidths=1.5, linestyles="-")

n_wedge = int(wedge.sum())
pct = 100 * n_wedge / wedge.size
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(MU_TICKS);  ax.set_xticklabels(MU_TICKS)
ax.set_yticks(LAM_TICKS); ax.set_yticklabels(LAM_TICKS)
ax.set_xlabel("firm mobility  $\\mu$", fontsize=11)
ax.set_ylabel("institutional adaptation  $\\lambda$", fontsize=11)
ax.set_title(f"Coordination-failure wedge\n"
             f"({n_wedge}/{wedge.size} = {pct:.0f}\\% of cells; each forgoes "
             f"$\\approx 575$\\,M of $W_{{\\mathrm{{tot}}}}$)", fontsize=11)

# legend patches
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#1a9850", label="green club (optimal)"),
                   Patch(facecolor="#d73027", label="race to bottom (suboptimal)")],
          loc="upper left", fontsize=9, framealpha=0.85)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"saved {OUT_PATH}")

# print summary stats for verification
print(f"\nW_tot by regime (from CSV):")
gc  = [r["all_W_tot"]/1e6 for r in rows if r["regime"] == "green_club"]
rtb = [r["all_W_tot"]/1e6 for r in rows if r["regime"] == "race_to_bottom"]
tr  = [r["all_W_tot"]/1e6 for r in rows if r["regime"] == "transitional"]
print(f"  green club:       n={len(gc)},  mean={np.mean(gc):.1f} M")
print(f"  race to bottom:   n={len(rtb)}, mean={np.mean(rtb):.1f} M")
print(f"  transitional:     n={len(tr)},  mean={np.mean(tr):.1f} M")
print(f"  wedge cells (RTB selected): {n_wedge} / {wedge.size} = {pct:.1f}%")
