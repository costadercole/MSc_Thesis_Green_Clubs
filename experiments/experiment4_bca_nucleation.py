"""
Experiment 4 — Does the border charge lower the founding threshold?

Experiment 3 showed a green club nucleates from a dirty start (h0=0.9) only if the
founding coalition exceeds a critical mass s0*. Here we add the border carbon
adjustment tau_BA as a swept axis and test whether it LOWERS that threshold — the
same lever that *sustains* a club should also make it cheaper to *found*.

Reuses the Experiment-3 nucleation harness exactly (sweep_lib): mu=lambda=nu=1,
dirty start h0=0.9, founding coalition = floor(s0*N) random strict jurisdictions,
relocation-only burn-in, then coupled dynamics; generic tariff held at tau=0.5.

Axes:  x = tau_BA in [0,1] (30 pts),  y = s0 in [0,1] (30 pts).
Validation: the tau_BA = 0.5 column must reproduce Experiment 3 (phi=1500): s0* ~ 0.86.

Run:
  python experiments/experiment4_bca_nucleation.py            # full 30x30, 6 seeds, T=2000
  python experiments/experiment4_bca_nucleation.py --quick    # coarse 12x12, 2 seeds, T=1000
Outputs: output/experiment_4/exp4_s0_tauBA_map.{png,csv},
         output/experiment_4/exp4_threshold_curve.png
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_lib import run_grid, save_csv, plot_grid, modal_regime, p_green, REGIME_GREEN, BASELINE

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "experiment_4")

H0_DIRTY = 0.9
TABLE_TAUBA = [0.0, 0.25, 0.5, 0.75, 1.0]


def s0_star_curve(res):
    """Per tau_BA column: smallest s0 from which the green club holds for all larger
    s0 (persistence rule, identical to Experiment 3)."""
    reg = modal_regime(res); xs = res["xvals"]; ys = res["yvals"]   # x=tau_BA, y=s0
    out = np.full(len(xs), np.nan)
    for j in range(len(xs)):
        green = reg[:, j] == REGIME_GREEN
        for i in range(len(ys)):
            if green[i:].all():
                out[j] = ys[i]; break
    return xs, out


def tipping_and_band(res):
    """Per tau_BA column: tipping s0 (mean final s crosses 0.5) and stochastic band."""
    xs, ys = res["xvals"], res["yvals"]
    S = np.nanmean(res["S"], axis=2)        # mean final s per cell (s0 rows, tauBA cols)
    pg = p_green(res)
    tip = np.full(len(xs), np.nan); band = np.full(len(xs), np.nan)
    for j in range(len(xs)):
        col_s = S[:, j]
        idx = np.where(col_s > 0.5)[0]
        tip[j] = ys[idx[0]] if len(idx) else np.nan
        contested = ys[(pg[:, j] > 0) & (pg[:, j] < 1)]
        band[j] = (contested.max() - contested.min()) if len(contested) else 0.0
    return xs, tip, band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    a = ap.parse_args()
    nx = a.nx or (12 if a.quick else 30)
    ny = a.ny or (12 if a.quick else 30)
    seeds = a.seeds or (2 if a.quick else 6)
    T = a.T or (1000 if a.quick else 2000)

    print(f"\nEXPERIMENT 4 — border charge x nucleation   grid {nx}(tauBA)×{ny}(s0), "
          f"{seeds} seeds, T={T}  (mu=lam=nu=1, h0={H0_DIRTY}, tau=0.5, phi={BASELINE.phi})")

    res = run_grid("tau_BA", np.linspace(0.0, 1.0, nx), "s0", np.linspace(0.0, 1.0, ny),
                   dict(mu=1.0, lam=1.0, nu=1.0, h0=H0_DIRTY, tau=0.5), seeds=seeds, T=T)
    save_csv(res, "exp4_s0_tauBA_map", outdir=OUTDIR)
    plot_grid(res, "exp4_s0_tauBA_map",
              "Founding a green club: outcome over border charge $\\tau_{BA}$ and initial coalition $s_0$",
              xlabel="$\\tau_{BA}$  (border carbon adjustment)",
              ylabel="$s_0$  (founding strict coalition)", outdir=OUTDIR)

    tauBA, s0star = s0_star_curve(res)
    _, tip, band = tipping_and_band(res)

    # validation: tau_BA = 0.5 column vs Experiment 3 (s0* ~ 0.86)
    j50 = int(np.argmin(np.abs(tauBA - 0.5)))
    print(f"\n[VALIDATION] s0*(tau_BA=0.5) = {s0star[j50]:.2f}  "
          f"(Experiment 3 phi=1500 gave s0* ~ 0.86)")

    # headline figure: s0*(tau_BA) + tipping midpoint
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(tauBA, s0star, "o-", color="#1a9850", lw=2, label="founding threshold $s_0^*$")
    ax.plot(tauBA, tip, "s--", color="0.45", lw=1.5, label="tipping midpoint $s_0$")
    ax.axvline(0.5, ls=":", color="0.6", lw=1, label="baseline $\\tau_{BA}=0.5$")
    ax.set_xlabel("$\\tau_{BA}$  (border carbon adjustment)")
    ax.set_ylabel("founding coalition $s_0$")
    ax.set_ylim(-0.03, 1.03); ax.set_title("Border charge lowers the founding threshold")
    ax.legend(loc="best", framealpha=0.9); fig.tight_layout()
    curve_path = os.path.join(OUTDIR, "exp4_threshold_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {curve_path}")

    # table at canonical tau_BA values
    print("\n  tau_BA | s0*  | tipping s0 | transition band")
    print("  -------|------|-----------|----------------")
    for tb in TABLE_TAUBA:
        j = int(np.argmin(np.abs(tauBA - tb)))
        ss = "—" if not np.isfinite(s0star[j]) else f"{s0star[j]:.2f}"
        tp = "—" if not np.isfinite(tip[j]) else f"{tip[j]:.2f}"
        print(f"   {tb:4.2f}  | {ss:>4} |   {tp:>5}    |   {band[j]:.2f}")

    # slope ds0*/dtau_BA over the finite, falling part
    m = np.isfinite(s0star)
    if m.sum() >= 2:
        slope = float(np.polyfit(tauBA[m], s0star[m], 1)[0])
        print(f"\n  slope ds0*/dtau_BA = {slope:.3f}")
        # tau_BA that brings s0* down to a modest majority (<=0.6)
        below = tauBA[m][s0star[m] <= 0.6]
        print(f"  tau_BA that brings s0* <= 0.6: "
              f"{'%.2f' % below.min() if len(below) else 'not reached in [0,1]'}")


if __name__ == "__main__":
    main()
