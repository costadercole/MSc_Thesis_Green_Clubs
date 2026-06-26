"""
Experiment B — Coalition nucleation (minimum viable club).

Can a green club nucleate from a lax-dominated, dirty start, and how large must the
initial committed coalition be?  Adds initial conditions as treatment variables:
  s0 = initial fraction of strict jurisdictions (floor(s0*N) nodes strict at t=0),
  h0 = initial high-emission share (default 0.9 — a dirty economy).
Both are existing params.Params fields wired into init_jurisdictions / init_firms;
each seed draws a fresh network, placement, AND random strict assignment.  The
relocation-only burn-in then sorts firms across the fixed initial policy landscape.

Primary map   : mu (log) × s0 (linear 0–1)   [lambda=nu=1, h0=0.9, instruments at baseline]
Secondary     : hysteresis line at mu=lambda=1 — fine s0 sweep, more seeds, final s vs s0.

Run:
  python experiments/experiment_min_club.py            # full 30×30×4, T=2000 + hysteresis
  python experiments/experiment_min_club.py --quick     # coarse 12×12×2, T=1000
Outputs: output/expB_s0_map.{npz,png}, output/expB_hysteresis.{npz,png},
         headline numbers appended to output/results_notes.md
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_lib import (run_grid, save_csv, plot_grid, modal_regime, p_green,
                       REGIME_GREEN, BASELINE)

MU_TICKS = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 20]
H0_DIRTY = 0.9


def mu_axis(n):
    return np.exp(np.linspace(np.log(0.02), np.log(20.0), n))


def s0_star_curve(res):
    """
    For each mu column, the smallest s0 *from which the green club holds for all
    larger s0* (a genuine critical mass is monotone — more initial coalition can
    only help). Requiring persistence up the column removes isolated single-seed
    green/non-green cells that a 'first green cell' rule would trip on.
    """
    reg = modal_regime(res)                # (ny=s0, nx=mu)
    s0 = res["yvals"]; mu = res["xvals"]
    out = np.full(len(mu), np.nan)
    for j in range(len(mu)):
        green = reg[:, j] == REGIME_GREEN  # over s0 ascending
        for i in range(len(s0)):
            if green[i:].all():
                out[j] = s0[i]
                break
    return mu, out


# ── Primary map: mu × s0 ────────────────────────────────────────────────────────
def run_primary(nx, ny, seeds, T, phi, delta_loc, suffix=""):
    res = run_grid("mu", mu_axis(nx), "s0", np.linspace(0.0, 1.0, ny),
                   dict(lam=1.0, nu=1.0, h0=H0_DIRTY, phi=phi, delta_loc=delta_loc),
                   seeds=seeds, T=T)
    tag = f"expB_s0_map{suffix}"
    save_csv(res, tag)
    plot_grid(res, tag,
              f"Coalition nucleation: outcome over mobility $\\mu$ and initial club size $s_0$"
              f"   ($\\varphi={phi:g}$)",
              xlabel="$\\mu$  (firm mobility = $\\mu/\\lambda$)",
              ylabel="$s_0$  (initial strict fraction)",
              xlog=True, xticks=MU_TICKS)
    return res


# ── Secondary: hysteresis line at mu = lambda = 1 ───────────────────────────────
def run_hysteresis(n_s0, seeds, T, phi, delta_loc, suffix=""):
    # 1-D sweep: x = s0 (n_s0 points), y = a single dummy axis (lam=1)
    res = run_grid("s0", np.linspace(0.0, 1.0, n_s0), "lam", np.array([1.0]),
                   dict(mu=1.0, nu=1.0, h0=H0_DIRTY, phi=phi, delta_loc=delta_loc),
                   seeds=seeds, T=T)
    save_csv(res, f"expB_hysteresis{suffix}")
    s0 = res["xvals"]
    s_mean = np.nanmean(res["S"][0], axis=1)            # mean final s per s0
    s_all  = res["S"][0]                                # (n_s0, seeds)
    pg     = p_green(res)[0]                            # fraction of seeds green per s0

    # tipping s0: where mean final s first exceeds 0.5
    tip = next((s0[i] for i in range(len(s0)) if s_mean[i] > 0.5), float("nan"))
    # transition band: s0 range where seeds disagree (0 < p_green < 1)
    band = s0[(pg > 0) & (pg < 1)]
    band_w = (float(band.max() - band.min()) if len(band) else 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for k in range(s_all.shape[1]):
        ax.plot(s0, s_all[:, k], color="0.8", lw=0.8, zorder=1)
    ax.plot(s0, s_mean, "o-", color="#1a9850", lw=2, zorder=3, label="mean final $s$")
    if np.isfinite(tip):
        ax.axvline(tip, ls="--", color="0.3", label=f"tipping $s_0\\approx{tip:.2f}$")
    ax.set_xlabel("$s_0$  (initial strict fraction)")
    ax.set_ylabel("steady-state strict fraction  $s$")
    ax.set_title(f"Hysteresis / tipping at $\\mu=\\lambda=1$ (dirty start $h_0=0.9$, $\\varphi={phi:g}$)")
    ax.set_ylim(-0.03, 1.03); ax.legend(loc="lower right", framealpha=0.9)
    out = f"output/expB_hysteresis{suffix}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return s0, s_mean, tip, band_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--which", choices=["both", "map", "hyst"], default="both")
    ap.add_argument("--phi", type=float, default=BASELINE.phi,
                    help="structural anchor: host benefit. Default=1500 (on boundary). "
                         "Use ~1350 for a green-side nucleation run with a lower, mu-dependent s0*.")
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
    contested = a.phi >= 1500
    suffix = "" if contested else f"_phi{int(round(a.phi))}"
    sec_key = "EXP_B" if contested else f"EXP_B_GREEN_{int(round(a.phi))}"
    anchor = "ON the structural boundary (contested)" if contested else "GREEN-SIDE of the boundary"
    print(f"\nEXPERIMENT B — nucleation   grid {nx}×{ny}, {seeds} seeds, T={T}  "
          f"(lambda=nu=1, h0={H0_DIRTY}, phi={a.phi}, delta_loc={a.delta_loc} — {anchor})")

    res_lines = ["## Experiment B — coalition nucleation (minimum viable club)\n",
                 f"Held fixed: lambda=nu=1, h0={H0_DIRTY} (dirty start), instruments tau=tau_BA=0.5, "
                 f"phi={a.phi}, delta_loc={a.delta_loc} ({anchor}). mu log-spaced 0.02–20.\n"]

    if a.which in ("both", "map"):
        print("\n[primary] mu × s0")
        res = run_primary(nx, ny, seeds, T, a.phi, a.delta_loc, suffix)
        mu, s0star = s0_star_curve(res)
        res_lines.append("\n### Minimum viable club  s0*(mu)")
        res_lines.append("| mu = (mu/lambda) | s0* (smallest green-club coalition) |")
        res_lines.append("|---|---|")
        for m, s in zip(mu, s0star):
            res_lines.append(f"| {m:.3f} | {'—' if not np.isfinite(s) else f'{s:.2f}'} |")
        finite = np.isfinite(s0star)
        if finite.any():
            res_lines.append(f"- at mu/lambda=1: s0* ≈ "
                             f"{_at(mu, s0star, 1.0)}; at mu/lambda=3: s0* ≈ {_at(mu, s0star, 3.0)}")

    if a.which in ("both", "hyst"):
        print("\n[secondary] hysteresis at mu=lambda=1")
        n_s0 = 12 if a.quick else 25
        h_seeds = (3 if a.quick else 10)
        s0, s_mean, tip, band_w = run_hysteresis(n_s0, h_seeds, T, a.phi, a.delta_loc, suffix)
        res_lines.append(f"\n### Tipping (hysteresis) at mu=lambda=1, {h_seeds} seeds")
        res_lines.append(f"- tipping s0 (mean final s crosses 0.5): **{tip:.2f}**" if np.isfinite(tip)
                         else "- tipping s0: not reached in [0,1]")
        res_lines.append(f"- stochastic transition-band width (seeds disagree): **{band_w:.2f}** in s0")

    # Headline numbers go to stdout → captured in the SLURM .out log (no md file).
    print("\n".join(res_lines))


def _at(mu, s0star, target):
    """s0* at the mu closest to target (only over finite entries)."""
    j = int(np.argmin(np.abs(mu - target)))
    v = s0star[j]
    return "—" if not np.isfinite(v) else f"{v:.2f} (mu={mu[j]:.2f})"


if __name__ == "__main__":
    main()
