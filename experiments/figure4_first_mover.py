"""
Figure 4 — distributional / first-mover dynamics.

Tracks the per-capita welfare of the FOUNDING strict bloc (the jurisdictions that
are strict at t=0) versus the initially-lax jurisdictions, over time, for a
nucleation that SUCCEEDS (s0 above threshold) and one that FAILS (below).

This makes the "punishment for going first" explicit: the founding bloc's welfare
dips while its dirty firms flee, before — if the coalition is large enough — the
club tips and welfare recovers above the lax level.

Reuses model.simulation.run (dynamics) and analysis.welfare.welfare_components
(eq. 3.42). Welfare is recomputed by re-solving the market at each recorded step.

Run: python experiments/figure4_first_mover.py [--outdir DIR]
"""

import os, sys, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from params import BASELINE
from model.simulation import run
from model.market import solve_market
from analysis.welfare import welfare_components

# contested baseline, diagnostic speed point (mu = lambda = nu = 1), dirty start
FIXED = dict(delta_loc=1000.0, delta_glob=250.0, phi=1500.0,
             mu=1.0, lam=1.0, nu=1.0, h0=0.9)
STRIDE = 4   # recompute welfare every STRIDE steps (steady-ish, keeps cost low)


def trajectory(s0, seed=42, T=2000):
    p = replace(BASELINE, s0=s0, T=T, seed=seed, **FIXED)
    res = run(p)
    P, W = res["P"], res["W"]
    founders = res["sigma"][0] == 1          # strict at t=0 (the founding coalition)
    lax0 = ~founders
    t, wf, wl, sfrac = [], [], [], []
    for step in range(0, T, STRIDE):
        fH, fL, sig = res["f_H"][step], res["f_L"][step], res["sigma"][step]
        ps, qH, qL = solve_market(fH, fL, sig, P, W, p)
        comp = welfare_components(fH, fL, sig, P, ps, qH, qL, W, p)
        wpc = comp["W_pc"]
        t.append(step * p.dt)
        wf.append(float(np.mean(wpc[founders])))
        wl.append(float(np.mean(wpc[lax0])))
        sfrac.append(float(res["s"][step]))
    return (np.array(t), np.array(wf), np.array(wl), np.array(sfrac),
            float(founders.mean()))


# Benchmark welfare levels from the speed-map (exp1_speed_map.csv averages)
W_GREEN = 16_992   # mean W/P in green-club cells
W_RTB   =  6_504   # mean W/P in race-to-bottom cells

COL_FOUND = "#1a9850"   # green  — founding bloc
COL_JOIN  = "#d73027"   # red    — late joiners
COL_S     = "#4d4d4d"   # dark grey — strict fraction line


def panel(ax, s0, seed, success: bool):
    t, wf, wl, s, s0real = trajectory(s0, seed)

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(W_GREEN, color=COL_FOUND, lw=0.9, ls=":", alpha=0.55, zorder=1)
    ax.axhline(W_RTB,   color=COL_JOIN,  lw=0.9, ls=":", alpha=0.55, zorder=1)
    ax.text(t[-1]*0.98, W_GREEN*1.01, "green-club\naverage",
            ha="right", va="bottom", fontsize=7.5, color=COL_FOUND, alpha=0.8)
    ax.text(t[-1]*0.98, W_RTB*1.02, "race-to-bottom\naverage",
            ha="right", va="bottom", fontsize=7.5, color=COL_JOIN, alpha=0.8)

    # ── strict fraction on secondary y-axis ───────────────────────────────────
    axR = ax.twinx()
    axR.plot(t, s, color=COL_S, lw=1.1, ls="--", alpha=0.5, zorder=1,
             label="strict fraction $s(t)$")
    axR.set_ylim(-0.05, 1.25)
    axR.set_ylabel("strict fraction $s(t)$  [dashed]", color=COL_S, fontsize=9)
    axR.tick_params(axis="y", colors=COL_S, labelsize=8)
    axR.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # ── welfare trajectories ──────────────────────────────────────────────────
    ax.plot(t, wf, color=COL_FOUND, lw=2.4, zorder=3,
            label="founding bloc  (strict at $t=0$)")
    ax.plot(t, wl, color=COL_JOIN,  lw=2.4, zorder=3,
            label="late joiners  (lax at $t=0$)")

    # annotate start of founders (founding cost)
    ax.scatter([t[0]], [wf[0]], color=COL_FOUND, s=55, zorder=5)
    ax.annotate(f"founding cost\n$W/P={wf[0]:,.0f}$",
                xy=(t[0], wf[0]), xytext=(t[-1]*0.08, wf[0]*0.72),
                fontsize=8, color=COL_FOUND,
                arrowprops=dict(arrowstyle="->", color=COL_FOUND, lw=1.1))

    # annotate final values
    ax.scatter([t[-1]], [wf[-1]], color=COL_FOUND, s=55, zorder=5)
    ax.scatter([t[-1]], [wl[-1]], color=COL_JOIN,  s=55, zorder=5)
    ax.annotate(f"$W/P={wf[-1]:,.0f}$",
                xy=(t[-1], wf[-1]), xytext=(t[-1]*0.82, wf[-1]*1.05),
                fontsize=8, color=COL_FOUND)
    ax.annotate(f"$W/P={wl[-1]:,.0f}$",
                xy=(t[-1], wl[-1]), xytext=(t[-1]*0.82, wl[-1]*1.05),
                fontsize=8, color=COL_JOIN)

    # first-mover penalty bracket (success panel only)
    if success and wl[-1] > wf[-1]:
        penalty = wl[-1] - wf[-1]
        x_br = t[-1] * 1.00
        ax.annotate("", xy=(x_br, wf[-1]), xytext=(x_br, wl[-1]),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.3))
        ax.text(x_br * 1.01, (wf[-1] + wl[-1]) / 2,
                f"first-mover\npenalty\n$\\Delta={penalty:,.0f}$",
                fontsize=8, va="center", color="0.3")

    # ── axes labels & title ───────────────────────────────────────────────────
    ax.set_xlabel("simulation time", fontsize=10)
    ax.set_ylabel("per-capita welfare  $W_i / P_i$", fontsize=10)
    ax.tick_params(labelsize=9)

    if success:
        title = (f"Successful nucleation  ($s_0 \\approx {s0real:.2f}$, "
                 f"$\\mu = \\lambda = 1$)\n"
                 "Founding bloc commits first, loses dirty firms, "
                 "but club tips and both blocs end better off")
    else:
        title = (f"Failed nucleation  ($s_0 \\approx {s0real:.2f}$, "
                 f"$\\mu = \\lambda = 1$)\n"
                 "Coalition below critical mass — club dissolves, "
                 "founders end worse than the race-to-bottom baseline")
    ax.set_title(title, fontsize=9.5, pad=8)

    # legend — combine main ax entries only
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left" if success else "lower right",
              fontsize=8.5, framealpha=0.9)

    return dict(s0=s0real, w_found_start=wf[0], w_found_final=wf[-1],
                w_join_final=wl[-1], s_final=s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--s0_success", type=float, default=0.95)
    ap.add_argument("--s0_fail", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    rs = panel(axes[0], a.s0_success, a.seed, success=True)
    rf = panel(axes[1], a.s0_fail,    a.seed, success=False)

    fig.suptitle(
        "First-mover dynamics at the contested baseline  "
        "($\\varphi=1500$,  $\\delta_{\\mathrm{loc}}=1000$,  "
        "$\\mu=\\lambda=1$,  dirty start $h_0=0.9$)",
        fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 1])

    out = os.path.join(a.outdir, "welfare_firstmover.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}\n")

    for tag, r in [("SUCCESS", rs), ("FAIL", rf)]:
        gap = r["w_join_final"] - r["w_found_final"]
        print(f"[{tag}] s0={r['s0']:.2f}  final s={r['s_final']:.2f}")
        print(f"   founders W/P:  start={r['w_found_start']:.0f}  "
              f"final={r['w_found_final']:.0f}  (climb={r['w_found_final']-r['w_found_start']:+.0f})")
        print(f"   late joiners final W/P={r['w_join_final']:.0f}  "
              f"first-mover penalty={gap:+.0f}\n")


if __name__ == "__main__":
    main()
