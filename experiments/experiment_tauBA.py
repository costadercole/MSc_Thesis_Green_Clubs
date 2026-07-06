"""
Experiment A — Border carbon adjustment as a policy lever (CBAM experiment).

Makes the border adjustment tau_BA (and, for contrast, the generic tariff tau) a
treatment variable and measures how far it shifts the green-club boundary in the
speed dimension.  Institutional speed is fixed at lambda = nu = 1, so the mobility
axis mu IS the speed ratio mu/lambda, and the critical mu* read off each row is the
critical ratio (mu/lambda)* that the lever buys.

Primary map   : mu (log) × tau_BA (linear)   [tau = 0.5 fixed]
Secondary map : mu (log) × tau    (linear)   [tau_BA = 0.5 fixed]

Everything else is the calibrated contested baseline (phi=1500, delta_loc=1000,
delta_glob=250, c_L=6, t=3, g=2.3) carried in params.BASELINE.

Run:
  python experiments/experiment_tauBA.py            # full 30×30×4, T=2000
  python experiments/experiment_tauBA.py --quick     # coarse 12×12×2, T=1000
Outputs: output/experiment_tauBA/expA_tauBA_map.{csv,png},
         output/experiment_tauBA/expA_tau_map.{csv,png};
         headline numbers print to stdout.
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_lib import run_grid, save_csv, plot_grid, mean_R, BASELINE

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "experiment_tauBA")

MU_TICKS = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 20]
TARGET_RATIOS = [1.0, 3.0]          # mu/lambda targets for "minimum effective lever"


def mu_axis(n):
    return np.exp(np.linspace(np.log(0.02), np.log(20.0), n))


def _smooth(v):
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(v, 1.0)
    except Exception:
        return v


def _ascending_crossing(coord, value, target=0.0):
    """Smallest coord at which `value` rises through `target` (value increasing).
    NaN if never above target; coord[0] if already above at the bottom."""
    v = np.asarray(value, float) - target
    idx = np.where(v > 0)[0]
    if len(idx) == 0:
        return np.nan
    if idx[0] == 0:
        return float(coord[0])
    i = idx[0]
    c0, c1 = coord[i-1], coord[i]; y0, y1 = v[i-1], v[i]
    return float(c0 - y0 * (c1 - c0) / (y1 - y0))


def extract_boundary(res):
    """
    lever*(mu): the critical lever value at each mobility mu — the smallest tau_BA
    (or tau) at which the cell turns green (R rises through 0 along the lever axis).
    This is the policy-relevant reading ("how much border adjustment is needed at
    mobility mu") and is robust: R is monotone increasing in the lever, unlike the
    mu axis which is non-monotone because of the re-greening at extreme mu.
    """
    mu = res["xvals"]; lever = res["yvals"]; R = mean_R(res)
    levstar = [_ascending_crossing(lever, _smooth(R[:, j])) for j in range(len(mu))]
    return mu, np.array(levstar, float)


def at_ratio(mu, levstar, target):
    """Critical lever at mu = target (nearest grid column)."""
    j = int(np.argmin(np.abs(mu - target)))
    return levstar[j], mu[j]


def per_unit_slope(mu, levstar):
    """Slope d(lever*)/d(log10 mu): extra lever needed per decade of mobility,
    over the rising part of the boundary (finite entries)."""
    m = np.isfinite(levstar)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log10(mu[m]), levstar[m], 1)[0])


def run_map(lever_name, lever_vals, fixed, nx, seeds, T, tag, title):
    res = run_grid("mu", mu_axis(nx), lever_name, lever_vals, fixed, seeds=seeds, T=T)
    save_csv(res, tag, outdir=OUTDIR)
    plot_grid(res, tag, title, xlabel="$\\mu$  (firm mobility = $\\mu/\\lambda$)",
              ylabel=("$\\tau_{BA}$  (border adjustment)" if lever_name == "tau_BA"
                      else "$\\tau$  (generic tariff)"), outdir=OUTDIR,
              xlog=True, xticks=MU_TICKS, hline=0.5, hline_label="baseline = 0.5")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse 12×12×2, T=1000")
    ap.add_argument("--which", choices=["both", "tau_ba", "tau"], default="both")
    ap.add_argument("--tauba_max", type=float, default=1.5)   # probe showed the switch is in [0, ~0.7]
    ap.add_argument("--tau_max",   type=float, default=1.5)
    ap.add_argument("--phi", type=float, default=BASELINE.phi,
                    help="structural anchor: host benefit (default = calibrated baseline 1500)")
    ap.add_argument("--delta_loc", type=float, default=BASELINE.delta_loc,
                    help="structural anchor: local damage (default = calibrated baseline 1000)")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    a = ap.parse_args()

    nx = a.nx or (12 if a.quick else 30)
    ny = a.ny or (12 if a.quick else 30)
    seeds = a.seeds or (2 if a.quick else 4)
    T = a.T or (1000 if a.quick else 2000)
    print(f"\nEXPERIMENT A — CBAM   grid {nx}×{ny}, {seeds} seeds, T={T}  "
          f"(lambda=nu=1, baseline phi={BASELINE.phi}, delta_loc={BASELINE.delta_loc})")

    out = {}
    if a.which in ("both", "tau_ba"):
        print("\n[primary] mu × tau_BA  (tau=0.5 fixed)")
        res = run_map("tau_BA", np.linspace(0.0, a.tauba_max, ny),
                      dict(lam=1.0, nu=1.0, tau=0.5, phi=a.phi, delta_loc=a.delta_loc),
                      nx, seeds, T, "expA_tauBA_map",
                      "CBAM: regulatory outcome over mobility $\\mu$ and border adjustment $\\tau_{BA}$")
        out["tau_BA"] = extract_boundary(res)

    if a.which in ("both", "tau"):
        print("\n[secondary] mu × tau  (tau_BA=0.5 fixed)")
        res = run_map("tau", np.linspace(0.0, a.tau_max, ny),
                      dict(lam=1.0, nu=1.0, tau_BA=0.5, phi=a.phi, delta_loc=a.delta_loc),
                      nx, seeds, T, "expA_tau_map",
                      "Generic tariff: regulatory outcome over mobility $\\mu$ and tariff $\\tau$")
        out["tau"] = extract_boundary(res)

    _report(out)


def _report(out):
    lines = ["## Experiment A — CBAM (border adjustment as a policy lever)\n",
             f"Held fixed: lambda=nu=1, phi={BASELINE.phi}, delta_loc={BASELINE.delta_loc}, "
             f"delta_glob={BASELINE.delta_glob}, c_L={BASELINE.c_L}, t={BASELINE.t}, "
             f"g={BASELINE.g}. mu log-spaced 0.02–20 (= mu/lambda since lambda=1).\n"
             "Reported as the critical lever lever*(mu): the smallest border adjustment "
             "(tariff) needed at mobility mu to secure the green club.\n"]
    levels = {}
    for lever_name, (mu, levstar) in out.items():
        sym = "tau_BA" if lever_name == "tau_BA" else "tau"
        levels[sym] = (mu, levstar)
        lines.append(f"\n### Lever: {sym}   —   critical {sym}*(mu)")
        lines.append(f"| mu = (mu/lambda) | {sym}* (smallest value giving green club) |")
        lines.append("|---|---|")
        for m, ls in zip(mu, levstar):
            if not np.isfinite(ls):
                cell = "not reached in range"
            elif ls <= 0:
                cell = "0 (green even at lever=0)"
            else:
                cell = f"{ls:.3f}"
            lines.append(f"| {m:.3f} | {cell} |")
        for tgt in TARGET_RATIOS:
            v, mloc = at_ratio(mu, levstar, tgt)
            lines.append(f"- minimum effective {sym} to secure the green club at "
                         f"mu/lambda = {tgt:g}: **{'not reached' if not np.isfinite(v) else f'{v:.2f}'}** "
                         f"(read at mu={mloc:.2f})")
        sl = per_unit_slope(mu, levstar)
        lines.append(f"- per-unit cost of mobility: d{sym}*/d(log10 mu) = **{sl:.3f}** "
                     f"(extra {sym} per decade of mobility)")
    if "tau_BA" in levels and "tau" in levels:
        mu = levels["tau_BA"][0]
        diff = np.nanmean(levels["tau"][1] - levels["tau_BA"][1])
        lines.append(f"\n**Carbon-targeted vs generic:** averaged across mu, the generic tariff "
                     f"needs {diff:+.2f} more units than the border adjustment to secure the green "
                     f"club (positive ⇒ tau_BA is the more efficient instrument per unit).")
    # Headline numbers go to stdout → captured in the SLURM .out log (no md file).
    print("\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
