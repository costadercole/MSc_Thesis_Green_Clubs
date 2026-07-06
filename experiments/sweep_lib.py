"""
Shared machinery for the policy phase-diagram experiments (Experiments A and B).

Reuses the Experiment-1 method exactly:
  - the full ABM (model.simulation.run) on top of the calibrated params.BASELINE,
  - steady state = mean of (h, s) over the final 20% of T steps,
  - the SAME box-rule regime classifier and order parameter R = s - h,
  - 4 seeds per cell by default, fresh ER network + firm placement (+ fresh random
    strict assignment) per seed.

It adds what the policy experiments need on top of phase_diagram.py:
  - persistence of the PER-SEED (h, s) for every cell (npz), so figures and headline
    numbers regenerate without rerunning the ABM,
  - a generic two-panel plot (regime + R with black zero contour) matching Exp 1,
  - a boundary-interpolation helper (where R crosses zero along an axis).

The model core and phase_diagram.py are left untouched.
"""

import sys, os, csv, time
from multiprocessing import Pool, cpu_count
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state
from analysis.welfare import steady_state_welfare

# welfare aggregates stored per cell (seed-averaged); see analysis/welfare.py
WELFARE_KEYS = ["all_W_tot", "all_cs_tot", "all_fisc_tot", "all_host_tot",
                "all_dmg_tot", "all_dmg_loc_tot", "all_dmg_glob_tot", "all_W_pc",
                "strict_W_tot", "strict_W_pc", "strict_n", "strict_pop",
                "lax_W_tot", "lax_W_pc", "lax_n", "lax_pop"]

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

# ── Regime classifier — identical thresholds to Experiment 1 ────────────────────
GREEN_H_HI, GREEN_S_LO = 0.33, 0.67
RTB_H_LO,   RTB_S_HI   = 0.67, 0.33
REGIME_RTB, REGIME_TRANS, REGIME_GREEN = 0, 1, 2
REGIME_NAMES = {0: "race_to_bottom", 1: "transitional", 2: "green_club"}
REGIME_LABELS = ["race-to-bottom", "transitional /\ncontested", "green club"]


def coarse_regime(h, s):
    if h < GREEN_H_HI and s > GREEN_S_LO:
        return REGIME_GREEN
    if h > RTB_H_LO and s < RTB_S_HI:
        return REGIME_RTB
    return REGIME_TRANS


# ── Parallel grid sweep (keeps per-seed h, s) ───────────────────────────────────
_CTX = {}

def _init(xname, yname, fixed, T, seeds, seed0):
    _CTX.update(xname=xname, yname=yname, fixed=fixed, T=T, seeds=seeds, seed0=seed0)


def _cell(args):
    ix, xv, iy, yv = args
    over = dict(_CTX["fixed"])
    over[_CTX["xname"]] = xv
    over[_CTX["yname"]] = yv
    if over.get("nu", "x") is None:                     # nu tracks lam if requested
        over["nu"] = over.get("lam", BASELINE.lam)
    if "delta_loc" in over:                             # keep delta_glob = delta_loc/4
        over["delta_glob"] = over["delta_loc"] / 4.0
    hs, ss, wel = [], [], []
    for sd in range(_CTX["seed0"], _CTX["seed0"] + _CTX["seeds"]):
        p_cell = replace(BASELINE, T=_CTX["T"], seed=sd, **over)
        res = run(p_cell)
        st = steady_state(res)
        hs.append(st["h_ss"]); ss.append(st["s_ss"])
        wel.append(steady_state_welfare(res, p_cell))   # welfare aggregates this seed
    return ix, iy, np.array(hs), np.array(ss), wel


def _fmt_time(s):
    s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{sec:02d}s"


def _progress_bar(k, total, t0, width=40):
    frac = k / total
    bar = "█" * int(width * frac) + "░" * (width - int(width * frac))
    el = time.time() - t0
    eta = el / k * (total - k) if k else 0
    rate = k / el if el > 0 else 0
    sys.stdout.write(f"\r  [{bar}] {frac*100:5.1f}%  {k}/{total} cells "
                     f"| {rate:.1f} cell/s | elapsed {_fmt_time(el)} | ETA {_fmt_time(eta)}   ")
    sys.stdout.flush()


def run_grid(xname, xvals, yname, yvals, fixed, seeds=4, T=2000, seed0=42):
    """Sweep the (xvals × yvals) grid; return a dict with per-seed H, S arrays."""
    xvals = np.asarray(xvals, float); yvals = np.asarray(yvals, float)
    nx, ny = len(xvals), len(yvals)
    H = np.full((ny, nx, seeds), np.nan)
    S = np.full((ny, nx, seeds), np.nan)
    Wel = {k: np.full((ny, nx), np.nan) for k in WELFARE_KEYS}   # seed-mean per cell
    tasks = [(ix, xv, iy, yv)
             for iy, yv in enumerate(yvals) for ix, xv in enumerate(xvals)]

    print(f"  sweep {xname}({nx}) × {yname}({ny}) × {seeds} seeds, T={T} "
          f"→ {len(tasks)*seeds} ABM runs on {N_WORKERS} workers  (+ welfare logging)")
    sys.stdout.flush()
    total = len(tasks)
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init,
              initargs=(xname, yname, fixed, T, seeds, seed0)) as pool:
        for k, (ix, iy, h, s, wel) in enumerate(pool.imap_unordered(_cell, tasks), 1):
            H[iy, ix] = h; S[iy, ix] = s
            for key in WELFARE_KEYS:
                Wel[key][iy, ix] = np.nanmean([w[key] for w in wel])
            _progress_bar(k, total, t0)
    print()
    print(f"  done in {_fmt_time(time.time()-t0)}")
    return dict(xname=xname, yname=yname, xvals=xvals, yvals=yvals,
                H=H, S=S, welfare=Wel, seeds=seeds, T=T, seed0=seed0, fixed=fixed)


# ── Aggregates ──────────────────────────────────────────────────────────────────
def mean_R(res):
    """Mean order parameter R = s - h over seeds, per cell."""
    return np.nanmean(res["S"] - res["H"], axis=2)

def modal_regime(res):
    ny, nx, ns = res["H"].shape
    out = np.zeros((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            regs = [coarse_regime(res["H"][i, j, k], res["S"][i, j, k]) for k in range(ns)]
            out[i, j] = int(np.bincount(regs, minlength=3).argmax())
    return out

def p_green(res):
    """Fraction of seeds landing in the green-club box, per cell."""
    ny, nx, ns = res["H"].shape
    out = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            out[i, j] = np.mean([coarse_regime(res["H"][i, j, k], res["S"][i, j, k]) == REGIME_GREEN
                                 for k in range(ns)])
    return out


# ── Persistence — a single CSV of the data per map (no npz / no meta) ───────────
def save_csv(res, name, outdir="output"):
    """
    Write one CSV with the full data for a map: one row per cell, with the
    aggregate (modal regime, R, mean h, mean s, fraction green) and the raw
    per-seed (h, s). This is the only data file produced.
    """
    os.makedirs(outdir, exist_ok=True)
    xs, ys = res["xvals"], res["yvals"]
    H, S, ns = res["H"], res["S"], res["seeds"]
    R, reg, pg = mean_R(res), modal_regime(res), p_green(res)
    Wel = res.get("welfare", {})
    wkeys = [k for k in WELFARE_KEYS if k in Wel]
    path = os.path.join(outdir, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        hdr = [res["xname"], res["yname"], "regime", "R", "h_mean", "s_mean", "p_green"]
        hdr += wkeys
        hdr += [f"h_seed{k}" for k in range(ns)] + [f"s_seed{k}" for k in range(ns)]
        w.writerow(hdr)
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                row = [f"{xv:.6g}", f"{yv:.6g}", REGIME_NAMES[reg[iy, ix]],
                       f"{R[iy, ix]:.4f}", f"{np.nanmean(H[iy, ix]):.4f}",
                       f"{np.nanmean(S[iy, ix]):.4f}", f"{pg[iy, ix]:.3f}"]
                row += [f"{Wel[k][iy, ix]:.6g}" for k in wkeys]
                row += [f"{H[iy, ix, k]:.4f}" for k in range(ns)]
                row += [f"{S[iy, ix, k]:.4f}" for k in range(ns)]
                w.writerow(row)
    print(f"  saved {path}")


# ── Boundary interpolation ──────────────────────────────────────────────────────
def crossing(coord, value, target=0.0, log=False):
    """First coord at which `value` crosses `target` (descending), linearly
    interpolated (in log-coord if log=True). NaN if no crossing."""
    coord = np.asarray(coord, float); value = np.asarray(value, float) - target
    idx = np.where(value < 0)[0]
    if len(idx) == 0 or idx[0] == 0:
        return np.nan
    i = idx[0]
    c0, c1 = coord[i-1], coord[i]; y0, y1 = value[i-1], value[i]
    if log:
        c0, c1 = np.log(c0), np.log(c1)
    xc = c0 - y0 * (c1 - c0) / (y1 - y0)
    return np.exp(xc) if log else xc


# ── Plotting — two-panel, Experiment-1 style ────────────────────────────────────
_CMAP = ListedColormap(["#b2182b", "#f4d03f", "#1a9850"])
_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _CMAP.N)


def _plain_log(ax, axis, ticks):
    if axis == "x":
        ax.set_xscale("log"); ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    else:
        ax.set_yscale("log"); ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


def plot_grid(res, name, title, xlabel, ylabel, outdir="output",
              xlog=False, ylog=False, xticks=None, yticks=None,
              hline=None, hline_label=None, smooth=1.0):
    xs, ys = res["xvals"], res["yvals"]
    reg, R = modal_regime(res), mean_R(res)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].pcolormesh(xs, ys, reg, cmap=_CMAP, norm=_NORM, shading="nearest")
    axes[0].set_title("Regime")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=_NORM, cmap=_CMAP), ax=axes[0], ticks=[0, 1, 2])
    cb.ax.set_yticklabels(REGIME_LABELS)

    pm = axes[1].pcolormesh(xs, ys, R, cmap="RdYlGn", vmin=-1, vmax=1, shading="nearest")
    axes[1].set_title("Order parameter  $R = s - h$")
    fig.colorbar(pm, ax=axes[1])

    # smoothed R=0 contour (display only)
    try:
        from scipy.ndimage import gaussian_filter
        Rc = gaussian_filter(np.nan_to_num(R, nan=0.0), sigma=smooth)
    except Exception:
        Rc = R
    X, Y = np.meshgrid(xs, ys)
    for ax in axes:
        try:
            ax.contour(X, Y, Rc, levels=[0.0], colors="black", linewidths=1.8)
        except Exception:
            pass
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if xlog: _plain_log(ax, "x", xticks if xticks is not None else xs)
        if ylog: _plain_log(ax, "y", yticks if yticks is not None else ys)
        if hline is not None:
            ax.axhline(hline, ls="--", color="0.25", lw=1, label=hline_label)
            if hline_label:
                ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
