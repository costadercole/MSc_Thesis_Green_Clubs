"""
2-D regime phase-diagram tool.

Sweeps any two named parameters on a grid, runs the ABM to steady state at
each cell (averaged over several seeds), classifies the outcome into one of
three coarse regimes, and renders:

  (1) a categorical phase map   — modal regime per cell
  (2) a continuous order map     — regime index  R = s_ss − h_ss
                                   (+1 = green club, −1 = race to bottom)
  (3) P(race-to-bottom) per cell — fraction of seeds landing in RTB
                                   (reveals the bistable / contested band)

Two ready-made presets
----------------------
  structural : phi  ×  delta_loc   — locates the green-club / RTB / contested
                                      regions in economic-fundamentals space.
  speed      : mu   ×  lam          — at a fixed *contested* structural point,
                                      shows how relative speed of firm flight
                                      vs policy imitation selects the basin.

Run:
  python experiments/phase_diagram.py structural
  python experiments/phase_diagram.py speed
  python experiments/phase_diagram.py speed --phi 560   # tune the structural point

Outputs: output/experiment_1/phase_<preset>.png  and  output/experiment_1/phase_<preset>.csv
"""

import sys, os, csv, time, argparse, itertools
from multiprocessing import Pool, cpu_count

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

OUTDIR = os.path.join(_ROOT, "output", "experiment_1")

# Coarse regime thresholds (collapse the 9-cell classifier to 3).
GREEN_H_HI = 0.33     # green club: h below this ...
GREEN_S_LO = 0.67     #            ... and s above this
RTB_H_LO   = 0.67     # race to bottom: h above this ...
RTB_S_HI   = 0.33     #                ... and s below this

REGIME_GREEN = 2
REGIME_TRANS = 1
REGIME_RTB   = 0
REGIME_NAMES = {REGIME_RTB: "race_to_bottom", REGIME_TRANS: "transitional", REGIME_GREEN: "green_club"}


def coarse_regime(h, s):
    if h < GREEN_H_HI and s > GREEN_S_LO:
        return REGIME_GREEN
    if h > RTB_H_LO and s < RTB_S_HI:
        return REGIME_RTB
    return REGIME_TRANS


# ─────────────────────────────────────────────────────────────────────────────
# Presets: (xaxis_name, x_values), (yaxis_name, y_values), fixed overrides
# ─────────────────────────────────────────────────────────────────────────────

N_GRID = 30   # points per axis (30×30 = 900 cells)

def make_preset(name, args):
    if name == "structural":
        # phi (x) × delta_loc (y). mu=lam fixed (ratio 1) so the map shows the
        # ECONOMIC fundamentals that set the regime, holding speed neutral.
        xs = ("phi",       list(np.linspace(0, 2500, N_GRID)))
        ys = ("delta_loc", list(np.linspace(200, 1500, N_GRID)))
        fixed = dict(mu=1.0, lam=1.0, nu=1.0)
        return xs, ys, fixed

    if name == "speed":
        # mu (x) × lam (y), log-spaced, at a fixed CONTESTED structural point.
        # phi chosen near the boundary so the basin is genuinely selectable.
        xs = ("mu",  list(np.exp(np.linspace(np.log(0.02), np.log(20.0), N_GRID))))
        ys = ("lam", list(np.exp(np.linspace(np.log(0.1),  np.log(5.0),  N_GRID))))
        fixed = dict(phi=args.phi, delta_loc=args.delta_loc, nu=None)  # nu tracks lam (set per cell)
        return xs, ys, fixed

    raise ValueError(f"unknown preset {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

_CTX = {}   # filled per process


def _init(xname, yname, fixed, T, n_seeds):
    _CTX.update(xname=xname, yname=yname, fixed=fixed, T=T, n_seeds=n_seeds)


def _cell(args):
    ix, xval, iy, yval = args
    over = dict(_CTX["fixed"])
    over[_CTX["xname"]] = xval
    over[_CTX["yname"]] = yval
    # nu=None sentinel => let firm-revision rate track lam (same timescale family)
    if over.get("nu", "x") is None:
        over["nu"] = over.get("lam", BASELINE.lam)
    # keep global damage pinned to local damage ratio used elsewhere
    if "delta_loc" in over:
        over["delta_glob"] = over["delta_loc"] / 4.0

    hs, ss, regs = [], [], []
    for sd in range(42, 42 + _CTX["n_seeds"]):
        p = replace(BASELINE, T=_CTX["T"], seed=sd, **over)
        sstate = steady_state(run(p))
        hs.append(sstate["h_ss"])
        ss.append(sstate["s_ss"])
        regs.append(coarse_regime(sstate["h_ss"], sstate["s_ss"]))
    regs = np.array(regs)
    # modal regime; P(RTB) over seeds; mean h,s
    modal = int(np.bincount(regs, minlength=3).argmax())
    p_rtb = float(np.mean(regs == REGIME_RTB))
    return ix, iy, modal, p_rtb, float(np.mean(hs)), float(np.mean(ss))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_preset(name, args):
    (xname, xvals), (yname, yvals), fixed = make_preset(name, args)
    nx, ny = len(xvals), len(yvals)

    print(f"\n  PHASE DIAGRAM — preset '{name}'  ({N_WORKERS} workers)")
    print(f"  x = {xname}: {nx} pts [{xvals[0]:.3g} … {xvals[-1]:.3g}]")
    print(f"  y = {yname}: {ny} pts [{yvals[0]:.3g} … {yvals[-1]:.3g}]")
    print(f"  fixed: {fixed}")
    print(f"  T={args.T}  seeds={args.seeds}  cells={nx*ny}  runs={nx*ny*args.seeds}")
    sys.stdout.flush()

    modal = np.zeros((ny, nx), dtype=int)
    p_rtb = np.zeros((ny, nx))
    h_map = np.zeros((ny, nx))
    s_map = np.zeros((ny, nx))

    tasks = [(ix, xv, iy, yv)
             for iy, yv in enumerate(yvals)
             for ix, xv in enumerate(xvals)]

    total = len(tasks)
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init,
              initargs=(xname, yname, fixed, args.T, args.seeds)) as pool:
        for k, (ix, iy, m, pr, hh, ss) in enumerate(pool.imap_unordered(_cell, tasks), 1):
            modal[iy, ix] = m
            p_rtb[iy, ix] = pr
            h_map[iy, ix] = hh
            s_map[iy, ix] = ss
            _progress_bar(k, total, t0)
    print()  # newline after the bar
    print(f"  done in {time.time()-t0:.0f}s")

    _save_csv(name, xname, yname, xvals, yvals, modal, p_rtb, h_map, s_map)
    _plot(name, xname, yname, xvals, yvals, modal, p_rtb, h_map, s_map, args)


def _fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{sec:02d}s"


def _progress_bar(k, total, t0, width=40):
    """Single-line progress bar: [####----] 45% | 1125/2500 | elapsed | ETA."""
    frac = k / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - t0
    eta = elapsed / k * (total - k) if k else 0
    rate = k / elapsed if elapsed > 0 else 0
    sys.stdout.write(
        f"\r  [{bar}] {frac*100:5.1f}%  {k}/{total} cells  "
        f"| {rate:.1f} cell/s | elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)}   "
    )
    sys.stdout.flush()


def _save_csv(name, xname, yname, xvals, yvals, modal, p_rtb, h_map, s_map):
    path = os.path.join(OUTDIR, f"phase_{name}.csv")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([xname, yname, "regime", "p_rtb", "h_ss", "s_ss"])
        for iy, yv in enumerate(yvals):
            for ix, xv in enumerate(xvals):
                w.writerow([f"{xv:.5g}", f"{yv:.5g}", REGIME_NAMES[modal[iy, ix]],
                            f"{p_rtb[iy, ix]:.3f}", f"{h_map[iy, ix]:.3f}", f"{s_map[iy, ix]:.3f}"])
    print(f"  saved {path}")


def _plot(name, xname, yname, xvals, yvals, modal, p_rtb, h_map, s_map, args):
    xv = np.array(xvals); yv = np.array(yvals)
    logx = name == "speed"
    logy = name == "speed"

    def extent_imshow(ax, data, cmap, norm=None, vmin=None, vmax=None):
        return ax.pcolormesh(xv, yv, data, cmap=cmap, norm=norm,
                             vmin=vmin, vmax=vmax, shading="nearest")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: categorical regime map
    cmap = ListedColormap(["#b2182b", "#f4d03f", "#1a9850"])  # RTB, trans, green
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    pm = extent_imshow(axes[0], modal, cmap, norm=norm)
    axes[0].set_title("Regime (modal over seeds)")
    cb = fig.colorbar(pm, ax=axes[0], ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["race-to-bottom", "transitional/\ncontested", "green club"])

    # Panel 2: order parameter R = s - h
    R = s_map - h_map
    pm2 = extent_imshow(axes[1], R, "RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Order parameter  R = s − h")
    fig.colorbar(pm2, ax=axes[1])

    # Panel 3: P(race-to-bottom) over seeds — highlights the bistable band
    pm3 = extent_imshow(axes[2], p_rtb, "magma", vmin=0, vmax=1)
    axes[2].set_title("P(race-to-bottom) over seeds")
    fig.colorbar(pm3, ax=axes[2])

    for ax in axes:
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")

    if name == "speed":
        sub = f"phi={args.phi}, delta_loc={args.delta_loc}  (contested structural point)"
    else:
        sub = "mu=lam (neutral speed); regime set by economic fundamentals"
    fig.suptitle(f"Phase diagram — {name}   [{sub}]", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"phase_{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=["structural", "speed"])
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--phi", type=float, default=1500.0,
                    help="speed preset: structural point phi (validated contested anchor)")
    ap.add_argument("--delta_loc", type=float, default=1000.0,
                    help="speed preset: structural point delta_loc (wide-gradient row)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    run_preset(args.preset, args)
