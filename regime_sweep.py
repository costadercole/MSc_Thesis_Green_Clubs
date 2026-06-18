"""
regime_sweep.py  —  Main results: can the green club emerge and persist?
------------------------------------------------------------------------
Central experiment of Chapter 5.

Research question
-----------------
Under what (mu/lambda, policy) combinations does the system converge to the
green-club outcome (h,s) -> (0,1) rather than the race-to-the-bottom (1,0)?

Method
------
Fix the calibrated baseline (c_H=4, c_L=6, t=5, g=2.4, a=20, b=1,
N=50, k=3, topology='er', M=500, T=1000, dt=1.0, relocate=True).

1. PRIMARY SWEEP  —  vary mu/lambda across [0.05, 5.0] (log-spaced, 14 pts),
   holding lambda=1.0 fixed and varying mu.  For each mu:
   - Run R=10 replications from an adverse initial condition
     (h0=0.5, s0=0.2: half the firms dirty, most jurisdictions lax).
   - Classify each run by s_final: GC (>0.8), RTB (<0.2), PH (in between).
   - Record GC frequency = share of runs that reach the green club.

2. POLICY SWEEP  —  repeat for 3 values of tau_BA (0, 5, 10) to show
   how the border carbon adjustment shifts the critical threshold (mu/lambda)*.

3. DAMAGE SWEEP  —  repeat for 3 values of delta_loc (500, 2000, 5000) to
   show how local environmental damage interacts with firm mobility.

The critical threshold (mu/lambda)* is the largest ratio at which GC
frequency is still >= 50%.  How it shifts across policy parameters is the
central comparative-statics result.

Outputs
-------
output/regime_sweep_result.json        raw numbers
output/regime_gc_freq_tau_BA.png       GC freq vs mu/lambda, lines by tau_BA
output/regime_gc_freq_delta_loc.png    GC freq vs mu/lambda, lines by delta_loc
output/regime_map_tau_BA.png           2-D regime map in (mu/lambda, tau_BA)
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from trajectory import run_trajectory

os.makedirs("output", exist_ok=True)

# ── Calibrated baseline (Table 4.2 + structural defaults) ─────────────────────
BASELINE = dict(
    c_H=4.0, c_L=6.0, t=5.0, tau=3.0, g=2.4,
    delta_loc=2000.0, delta_glob=500.0,
    tau_BA=5.0,
    kappa=1.0,
    N=50, k=3, topology="er", M=500,
    mu_P=7.0, sigma_P=1.5,
    T=1000, relocate=True,
)

# ── Sweep design ───────────────────────────────────────────────────────────────
LAM_FIXED    = 1.0
MU_VALUES    = np.logspace(np.log10(0.05), np.log10(5.0), 14)
MU_LAM_RATIO = MU_VALUES / LAM_FIXED          # since lam is fixed this equals mu

R_REPS = 10          # replications per cell (different seeds -> different networks)
H0     = 0.5         # initial share of high-emission firms
S0     = 0.2         # adverse start: most jurisdictions lax

GC_THRESHOLD  = 0.8
RTB_THRESHOLD = 0.2

TAU_BA_VALUES    = [0.0, 5.0, 10.0]
DELTA_LOC_VALUES = [500.0, 2000.0, 5000.0]


def classify(s_final: float) -> str:
    if s_final > GC_THRESHOLD:  return "GC"
    if s_final < RTB_THRESHOLD: return "RTB"
    return "PH"


# ── Parallel worker ────────────────────────────────────────────────────────────

def _worker(args):
    mu, lam, seed, kw = args
    _, _, _, s_final = run_trajectory(
        mu=mu, lam=lam, h0=H0, s0=S0, seed=seed, **kw
    )
    return s_final


def run_sweep(mu_values, lam, r_reps, extra_kw, label):
    """
    For each mu in mu_values run r_reps replications; return
    (gc_freq, rtb_freq, ph_freq, mean_s_final) as (n_mu,) arrays.
    """
    kw = {**BASELINE, **extra_kw}
    jobs = [
        (mu, lam, seed, kw)
        for mu in mu_values
        for seed in range(r_reps)
    ]
    t0 = time.time()
    print(f"  {label}: {len(jobs)} trajectories ... ", end="", flush=True)

    with Pool(processes=cpu_count()) as pool:
        s_finals = list(pool.imap(_worker, jobs, chunksize=4))

    elapsed = time.time() - t0
    print(f"done ({elapsed:.0f}s)")

    arr = np.array(s_finals).reshape(len(mu_values), r_reps)
    gc_freq  = np.mean(arr > GC_THRESHOLD,  axis=1)
    rtb_freq = np.mean(arr < RTB_THRESHOLD, axis=1)
    ph_freq  = 1.0 - gc_freq - rtb_freq
    mean_s   = np.mean(arr, axis=1)
    return gc_freq, rtb_freq, ph_freq, mean_s


def critical_threshold(mu_lam_ratio, gc_freq):
    """Largest mu/lambda at which GC frequency >= 50%, or None."""
    crossing = mu_lam_ratio[gc_freq >= 0.5]
    return float(crossing[-1]) if len(crossing) > 0 else None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("REGIME SWEEP  —  Green club emergence across mu/lambda")
    print(f"  mu grid ({len(MU_VALUES)} pts, log-spaced): "
          f"{MU_VALUES[0]:.3f} ... {MU_VALUES[-1]:.3f}")
    print(f"  lambda fixed = {LAM_FIXED}")
    print(f"  Initial condition: h0={H0}, s0={S0}  (adverse / mostly-lax start)")
    print(f"  R = {R_REPS} replications per cell   T = {BASELINE['T']} periods")
    print(f"  tau_BA sweep : {TAU_BA_VALUES}")
    print(f"  delta_loc sweep: {DELTA_LOC_VALUES}")
    print(f"  Cores available: {cpu_count()}")
    print("=" * 72)

    # ── Sweep 1: vary tau_BA ───────────────────────────────────────────────────
    print("\n[1/2] tau_BA sweep")
    tau_ba_results = {}
    for tba in TAU_BA_VALUES:
        gc, rtb, ph, ms = run_sweep(
            MU_VALUES, LAM_FIXED, R_REPS, {"tau_BA": tba}, f"tau_BA={tba:.0f}"
        )
        tau_ba_results[tba] = {
            "gc_freq": gc.tolist(), "rtb_freq": rtb.tolist(),
            "ph_freq": ph.tolist(), "mean_s": ms.tolist(),
            "critical_mu_lam": critical_threshold(MU_LAM_RATIO, gc),
        }

    # ── Sweep 2: vary delta_loc ────────────────────────────────────────────────
    print("\n[2/2] delta_loc sweep")
    delta_loc_results = {}
    for dloc in DELTA_LOC_VALUES:
        gc, rtb, ph, ms = run_sweep(
            MU_VALUES, LAM_FIXED, R_REPS, {"delta_loc": dloc}, f"delta_loc={dloc:.0f}"
        )
        delta_loc_results[dloc] = {
            "gc_freq": gc.tolist(), "rtb_freq": rtb.tolist(),
            "ph_freq": ph.tolist(), "mean_s": ms.tolist(),
            "critical_mu_lam": critical_threshold(MU_LAM_RATIO, gc),
        }

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "mu_values": MU_VALUES.tolist(),
        "mu_lam_ratio": MU_LAM_RATIO.tolist(),
        "lam_fixed": LAM_FIXED,
        "h0": H0, "s0": S0, "R": R_REPS,
        "baseline": BASELINE,
        "tau_ba_sweep": {str(k): v for k, v in tau_ba_results.items()},
        "delta_loc_sweep": {str(k): v for k, v in delta_loc_results.items()},
    }
    json_path = "output/regime_sweep_result.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print()
    print("Critical (mu/lambda)* where GC frequency crosses 50%:")
    for tba in TAU_BA_VALUES:
        c = tau_ba_results[tba]["critical_mu_lam"]
        print(f"  tau_BA={tba:.0f}:   (mu/lam)* = "
              + (f"{c:.3f}" if c else "< {:.3f} (GC never reaches 50%%)".format(MU_LAM_RATIO[0])))
    for dloc in DELTA_LOC_VALUES:
        c = delta_loc_results[dloc]["critical_mu_lam"]
        print(f"  delta_loc={dloc:.0f}: (mu/lam)* = "
              + (f"{c:.3f}" if c else "< {:.3f} (GC never reaches 50%%)".format(MU_LAM_RATIO[0])))

    # ── Plot 1: GC freq vs mu/lambda, lines by tau_BA ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    COLORS = ["#1565C0", "#EF6C00", "#2E7D32"]
    for (tba, res), col in zip(tau_ba_results.items(), COLORS):
        ax.plot(MU_LAM_RATIO, res["gc_freq"], "o-", color=col,
                label=f"$\\tau_{{BA}}$ = {tba:.0f}", lw=2.0, ms=6)
        c = res["critical_mu_lam"]
        if c:
            ax.axvline(c, color=col, ls="--", lw=1.0, alpha=0.5)

    ax.axhline(0.5, ls=":", color="grey", lw=1, label="50% threshold")
    ax.set_xscale("log")
    ax.set_xlabel("$\\mu / \\lambda$  (firm mobility / institutional adaptation speed)",
                  fontsize=11)
    ax.set_ylabel("Green-club emergence frequency", fontsize=11)
    ax.set_title(
        "Can the green club emerge?\n"
        f"Adverse start: $h_0={H0}$, $s_0={S0}$  |  "
        f"{R_REPS} replications  |  $T={BASELINE['T']}$",
        fontsize=11,
    )
    ax.legend(fontsize=10, title="Border carbon\nadjustment")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    p1 = "output/regime_gc_freq_tau_BA.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ── Plot 2: GC freq vs mu/lambda, lines by delta_loc ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    COLORS2 = ["#6A1B9A", "#C62828", "#00695C"]
    for (dloc, res), col in zip(delta_loc_results.items(), COLORS2):
        ax.plot(MU_LAM_RATIO, res["gc_freq"], "s-", color=col,
                label=f"$\\delta_{{loc}}$ = {dloc:.0f}", lw=2.0, ms=6)
        c = res["critical_mu_lam"]
        if c:
            ax.axvline(c, color=col, ls="--", lw=1.0, alpha=0.5)

    ax.axhline(0.5, ls=":", color="grey", lw=1, label="50% threshold")
    ax.set_xscale("log")
    ax.set_xlabel("$\\mu / \\lambda$  (firm mobility / institutional adaptation speed)",
                  fontsize=11)
    ax.set_ylabel("Green-club emergence frequency", fontsize=11)
    ax.set_title(
        "Environmental damage and green-club emergence\n"
        f"Adverse start: $h_0={H0}$, $s_0={S0}$  |  "
        f"{R_REPS} replications  |  $T={BASELINE['T']}$",
        fontsize=11,
    )
    ax.legend(fontsize=10, title="Local damage")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    p2 = "output/regime_gc_freq_delta_loc.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # ── Plot 3: regime map in (mu/lambda, tau_BA) space ───────────────────────
    gc_matrix = np.array([tau_ba_results[tba]["gc_freq"] for tba in TAU_BA_VALUES])

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(
        gc_matrix,
        origin="lower",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        aspect="auto",
        extent=[0, len(MU_VALUES), -0.5, len(TAU_BA_VALUES) - 0.5],
    )
    ax.set_xticks(np.arange(len(MU_VALUES)) + 0.5)
    ax.set_xticklabels([f"{r:.2f}" for r in MU_LAM_RATIO], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(TAU_BA_VALUES)))
    ax.set_yticklabels([f"{v:.0f}" for v in TAU_BA_VALUES], fontsize=10)
    ax.set_xlabel("$\\mu / \\lambda$", fontsize=11)
    ax.set_ylabel("$\\tau_{BA}$", fontsize=11)
    ax.set_title(
        "Regime map: green-club emergence frequency\n"
        f"(adverse start $s_0={S0}$, {R_REPS} reps)",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="GC frequency", shrink=0.8)
    fig.tight_layout()
    p3 = "output/regime_map_tau_BA.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
