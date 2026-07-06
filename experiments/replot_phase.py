"""
Polished re-plot of a phase diagram from its saved CSV — no re-simulation.

Reads output/phase_<preset>.csv and produces a thesis-ready figure:
  - left  : categorical regime map (green club / transitional / race-to-bottom)
  - right : order parameter R = s - h, with the R = 0 regime boundary overlaid
            as a smooth contour (this reads cleanly even at modest seed counts)
  - speed preset: log axes labelled with plain numbers (0.02, 0.1, 1, 10),
            and the neutral-speed line mu = lam drawn dashed.
  - structural preset: the calibrated anchor (phi=1500, delta_loc=1000) marked.

Run:
  python experiments/replot_phase.py speed
  python experiments/replot_phase.py structural
Outputs: output/experiment_1/phase_<preset>_final.png
"""

import sys, os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "experiment_1")

REGIME_TO_INT = {"race_to_bottom": 0, "transitional": 1, "green_club": 2}
REGIME_LABELS = ["race-to-bottom", "transitional /\ncontested", "green club"]

# Plain-number ticks for the log-scaled speed axes
MU_TICKS  = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 20]
LAM_TICKS = [0.1, 0.2, 0.5, 1, 2, 5]

ANCHOR = {"phi": 1500.0, "delta_loc": 1000.0}   # marked on the structural map


def load_grid(preset):
    path = os.path.join(OUTDIR, f"phase_{preset}.csv")
    rows = list(csv.DictReader(open(path)))
    xname, yname = list(rows[0].keys())[0], list(rows[0].keys())[1]
    xs = sorted({float(r[xname]) for r in rows})
    ys = sorted({float(r[yname]) for r in rows})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    reg = np.full((len(ys), len(xs)), np.nan)
    R   = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        i, j = yi[float(r[yname])], xi[float(r[xname])]
        reg[i, j] = REGIME_TO_INT[r["regime"]]
        R[i, j]   = float(r["s_ss"]) - float(r["h_ss"])
    return xname, yname, np.array(xs), np.array(ys), reg, R


def _plain_log_axis(ax, which, ticks):
    setter = ax.set_xticks if which == "x" else ax.set_yticks
    lbl    = ax.set_xticklabels if which == "x" else ax.set_yticklabels
    if which == "x":
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    else:
        ax.set_yscale("log"); ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    setter(ticks)
    lbl([f"{t:g}" for t in ticks])


def plot(preset):
    xname, yname, xs, ys, reg, R = load_grid(preset)
    is_speed = preset == "speed"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel 1: categorical regime ----
    cmap = ListedColormap(["#b2182b", "#f4d03f", "#1a9850"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    axes[0].pcolormesh(xs, ys, reg, cmap=cmap, norm=norm, shading="nearest")
    axes[0].set_title("Regime")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0], ticks=[0, 1, 2])
    cb.ax.set_yticklabels(REGIME_LABELS)

    # ---- Panel 2: order parameter R = s - h, + boundary contour ----
    pm = axes[1].pcolormesh(xs, ys, R, cmap="RdYlGn", vmin=-1, vmax=1, shading="nearest")
    axes[1].set_title("Order parameter  $R = s - h$")
    fig.colorbar(pm, ax=axes[1])

    # R = 0 boundary contour on both panels. Light Gaussian smoothing of R is
    # applied ONLY for the contour (the heatmap above still shows raw cell values),
    # so the boundary reads as a single clean curve despite seed-level noise.
    # The grid is uniform in index space (log-spaced for speed, linear for
    # structural), so a uniform-sigma filter is appropriate on either.
    X, Y = np.meshgrid(xs, ys)
    try:
        from scipy.ndimage import gaussian_filter
        sigma = 1.3 if is_speed else 0.6
        R_contour = gaussian_filter(np.nan_to_num(R, nan=0.0), sigma=sigma)
    except Exception:
        R_contour = R
    for ax in axes:
        try:
            ax.contour(X, Y, R_contour, levels=[0.0], colors="black", linewidths=1.8)
        except Exception:
            pass

    # ---- axes formatting ----
    for ax in axes:
        ax.set_xlabel("$\\mu$" if is_speed else "$\\varphi$  (host benefit)")
        ax.set_ylabel("$\\lambda$" if is_speed else "$\\delta_{loc}$  (local damage)")
        if is_speed:
            _plain_log_axis(ax, "x", MU_TICKS)
            _plain_log_axis(ax, "y", LAM_TICKS)
            # neutral-speed line mu = lam
            lo = max(xs.min(), ys.min()); hi = min(xs.max(), ys.max())
            ax.plot([lo, hi], [lo, hi], ls="--", color="0.3", lw=1, label="$\\mu=\\lambda$")
            ax.legend(loc="lower right", framealpha=0.85)
        else:
            ax.scatter([ANCHOR["phi"]], [ANCHOR["delta_loc"]], s=90, marker="*",
                       facecolor="white", edgecolor="black", zorder=5,
                       label="calibrated baseline")
            ax.legend(loc="upper left", framealpha=0.85)

    if is_speed:
        title = "Regulatory outcome by firm mobility $\\mu$ and institutional speed $\\lambda$"
    else:
        title = "Regulatory outcome by host benefit $\\varphi$ and local damage $\\delta_{loc}$"
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"phase_{preset}_final.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=["speed", "structural"])
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    plot(args.preset)
