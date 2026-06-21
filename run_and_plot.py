"""
Run one baseline simulation and save diagnostic plots to output/.

Usage
-----
  python run_and_plot.py                          # baseline parameters
  python run_and_plot.py --mu 0.5 --lam 2.0      # override mu and lam
  python run_and_plot.py --mu 3.0 --T 2000       # high mobility, longer run
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state, classify_outcome
from analysis.plots import plot_time_series, plot_phase_portrait, plot_firm_distribution

os.makedirs("output", exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mu",       type=float, default=None)
    p.add_argument("--lam",      type=float, default=None)
    p.add_argument("--tau_BA",   type=float, default=None)
    p.add_argument("--t",        type=float, default=None)
    p.add_argument("--g",        type=float, default=None)
    p.add_argument("--kappa",    type=float, default=None)
    p.add_argument("--T",        type=int,   default=None)
    p.add_argument("--seed",     type=int,   default=None)
    p.add_argument("--topology", type=str,   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    p      = replace(BASELINE, **kwargs)

    label = f"mu={p.mu:.2f}_lam={p.lam:.2f}_t={p.t:.1f}_tBA={p.tau_BA:.1f}"
    print(f"Running simulation: {label}")
    print(f"  T={p.T}  topology={p.topology}  k={p.k}  seed={p.seed}  relocate={p.relocate}")

    results = run(p)

    ss      = steady_state(results)
    outcome = classify_outcome(ss["h_ss"], ss["s_ss"])
    print(f"\nSteady state:  h = {ss['h_ss']:.3f} ± {ss['h_std']:.3f}")
    print(f"               s = {ss['s_ss']:.3f} ± {ss['s_std']:.3f}")
    print(f"Outcome: {outcome}")

    # --- Time series ---
    fig = plot_time_series(results, title=f"h(t) and s(t)  [{label}]")
    path = f"output/ts_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Phase portrait ---
    fig = plot_phase_portrait(results, title=f"Phase portrait  [{label}]")
    path = f"output/phase_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Firm distribution at final step ---
    fig = plot_firm_distribution(results, step=-1, title=f"Firm distribution (t={p.T})  [{label}]")
    path = f"output/firms_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
