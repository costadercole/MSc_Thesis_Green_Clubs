"""
Experiment 1 re-run THROUGH sweep_lib so the welfare layer is logged.

phase_diagram.py produced the original Exp-1 figures but does not log welfare.
This reruns the same two maps via sweep_lib (identical ranges, grid, seeds,
box-rule classification and R), which additionally records the per-cell welfare
aggregates (W_tot, four components, bloc split) into the CSV — needed for the
welfare maps and the selected-vs-optimal wedge analysis.

  speed      : mu (log 0.02–20) × lambda (log 0.1–5), at the contested point
               phi=1500, delta_loc=1000 (nu tracks lambda).
  structural : phi (0–2500) × delta_loc (200–1500), at neutral speed mu=lam=nu=1.

Run:
  python experiments/experiment1_welfare.py speed
  python experiments/experiment1_welfare.py structural
  python experiments/experiment1_welfare.py both
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_lib import run_grid, save_csv, plot_grid, BASELINE

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "experiment_welfare")

MU_TICKS  = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 20]
LAM_TICKS = [0.1, 0.2, 0.5, 1, 2, 5]


def run_speed(n, seeds, T):
    res = run_grid("mu",  np.exp(np.linspace(np.log(0.02), np.log(20.0), n)),
                   "lam", np.exp(np.linspace(np.log(0.1),  np.log(5.0),  n)),
                   dict(phi=1500.0, delta_loc=1000.0, nu=None), seeds=seeds, T=T)
    save_csv(res, "exp1_speed_map", outdir=OUTDIR)
    plot_grid(res, "exp1_speed_map",
              "Regulatory outcome by firm mobility $\\mu$ and institutional speed $\\lambda$",
              xlabel="$\\mu$", ylabel="$\\lambda$", outdir=OUTDIR,
              xlog=True, ylog=True, xticks=MU_TICKS, yticks=LAM_TICKS)


def run_structural(n, seeds, T):
    res = run_grid("phi",       np.linspace(0, 2500, n),
                   "delta_loc", np.linspace(200, 1500, n),
                   dict(mu=1.0, lam=1.0, nu=1.0), seeds=seeds, T=T)
    save_csv(res, "exp1_structural_map", outdir=OUTDIR)
    plot_grid(res, "exp1_structural_map",
              "Regulatory outcome by host benefit $\\varphi$ and local damage $\\delta_{loc}$",
              xlabel="$\\varphi$  (host benefit)", ylabel="$\\delta_{loc}$  (local damage)",
              outdir=OUTDIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=["speed", "structural", "both"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    a = ap.parse_args()
    n = a.n or (12 if a.quick else 30)
    seeds = a.seeds or (2 if a.quick else 4)
    T = a.T or (1000 if a.quick else 2000)
    if a.preset in ("speed", "both"):
        print("\n[Exp 1] speed map (mu × lambda) + welfare"); run_speed(n, seeds, T)
    if a.preset in ("structural", "both"):
        print("\n[Exp 1] structural map (phi × delta_loc) + welfare"); run_structural(n, seeds, T)


if __name__ == "__main__":
    main()
