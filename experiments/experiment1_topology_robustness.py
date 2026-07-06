"""
Robustness check for Experiment 1's speed phase diagram (mu x lambda) across
network topologies other than the reported Erdos-Renyi (k=3) baseline.

The ER speed map (output/experiment_1/phase_speed.csv ->
output/experiment_1/phase_speed_final.png) used
4 seeds/cell. This script reruns the identical mu x lambda sweep -- same grid,
same contested structural point (phi=1500, delta_loc=1000, delta_glob=250),
same nu=lam-per-cell rule, same T=2000, same box-rule classifier -- at 8
seeds/cell, on FOUR structured topologies that bracket the ER baseline's
realised mean degree (~3.3) from below (~2) and above (~4), so that at each
degree level the ring-vs-BA comparison is degree-matched and isolates the
effect of network *structure* (spatial order vs. hub/power-law) rather than
confounding it with a degree mismatch:

  ring_k2 : k-regular ring lattice, k=2 (Watts-Strogatz p=0)  -> degree 2 (exact)
  ring_k4 : k-regular ring lattice, k=4 (Watts-Strogatz p=0)  -> degree 4 (exact)
  ba_m1   : Barabasi-Albert, m=1 explicit                     -> mean degree ~2
  ba_m2   : Barabasi-Albert, m=2 explicit                     -> mean degree ~4

Also reruns the ER baseline itself at the same 8-seeds/cell protocol (label
"er") so the summary table is apples-to-apples across all five topologies.
The *reported* thesis figure (phase_speed_final.png) used 4 seeds/cell; this
script's ER row is a separate, independent recomputation and does NOT
overwrite the experiment_1 outputs (everything here goes to
output/experiment_topology/).

Two bugs in the original version of this script produced invalid output and
are fixed here:

  1. BA's m was derived from k via m = max(1, k // 2). That map is not
     injective at low k: k=1 and k=2 BOTH give m=1, so the old "ba1"
     (k=1) and "ba2" (k=2) configs silently built the SAME graph (m=1)
     twice. Fix: pass m explicitly (m=1, m=2) instead of deriving it from k.
  2. The ring was built as nx.watts_strogatz_graph(N, k, p=0) with k=3. WS
     with p=0 connects each node to k // 2 neighbours per side, so the
     REALISED degree is 2 * (k // 2) -- exact only for EVEN k. k=3 silently
     produced degree 2, not 3. Fix: only ever request ring degree with even
     k (k=2, k=4), which is also what lets ring and BA be degree-matched
     without doctoring either construction.

Guard against a repeat of either bug: before running the expensive sweep for
each topology, `sanity_check` builds the network, prints its realised mean
degree and connectivity, and ASSERTS the realised mean degree is within
+/-0.3 of the intended target -- failing loudly (raising, not warning) if
not, rather than silently sweeping over the wrong graph again.

Run (locally or on Snellius via the companion .sh scripts):
  python experiments/experiment1_topology_robustness.py ring
  python experiments/experiment1_topology_robustness.py ba
  python experiments/experiment1_topology_robustness.py all
"""

import os, sys, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_lib import run_grid, mean_R, modal_regime, p_green, crossing, REGIME_NAMES
from model.network import build_network

REGIME_LABELS = ["race-to-bottom", "transitional /\ncontested", "green club"]
_CMAP = ListedColormap(["#b2182b", "#f4d03f", "#1a9850"])
_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _CMAP.N)

MU_TICKS  = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 20]
LAM_TICKS = [0.1, 0.2, 0.5, 1, 2, 5]

DEGREE_TOL    = 0.3   # +/- tolerance for ring/BA (deterministic / tightly concentrated)
DEGREE_TOL_ER = 1.0   # +/- tolerance for ER (random graph; single-seed mean degree at
                       # N=50, k=3 has sd ~0.2-0.3, so 0.3 alone is too tight and would
                       # false-positive on ordinary sampling draws)


def _plain_log_axis(ax, which, ticks):
    setter = ax.set_xticks if which == "x" else ax.set_yticks
    lbl    = ax.set_xticklabels if which == "x" else ax.set_yticklabels
    if which == "x":
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    else:
        ax.set_yscale("log"); ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    setter(ticks)
    lbl([f"{t:g}" for t in ticks])


def plot_speed_map(res, out_path, title):
    """Two-panel figure in the exact phase_speed_final.png layout/style:
    regime map + R=s-h order parameter, R=0 contour, mu=lam neutral diagonal."""
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

    X, Y = np.meshgrid(xs, ys)
    try:
        from scipy.ndimage import gaussian_filter
        R_contour = gaussian_filter(np.nan_to_num(R, nan=0.0), sigma=1.3)
    except Exception:
        R_contour = R
    for ax in axes:
        try:
            ax.contour(X, Y, R_contour, levels=[0.0], colors="black", linewidths=1.8)
        except Exception:
            pass
        ax.set_xlabel("$\\mu$"); ax.set_ylabel("$\\lambda$")
        _plain_log_axis(ax, "x", MU_TICKS)
        _plain_log_axis(ax, "y", LAM_TICKS)
        lo = max(xs.min(), ys.min()); hi = min(xs.max(), ys.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="0.3", lw=1, label="$\\mu=\\lambda$")
        ax.legend(loc="lower right", framealpha=0.85)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


N_GRID_DEFAULT = 30
SEEDS_DEFAULT  = 8     # reported ER figure used 4; this script uses 8 throughout, ER included
T_DEFAULT      = 2000
PHI, DELTA_LOC, DELTA_GLOB = 1500.0, 1000.0, 250.0

# label -> (topology string for Params, k passed to Params, m passed explicitly to
#           build_network's sanity check [None for non-BA], intended mean degree, description)
#
# Note on "intended mean degree" for ER: ER's realised mean degree is a random
# variable with mean k but nonzero sampling variance at finite N (at N=50, k=3
# the single-seed sd is roughly 0.2-0.3), so the *target* here is the true
# expectation k=3, not any single seed's realised value -- the tolerance for
# ER (DEGREE_TOL_ER below) is widened accordingly so ordinary sampling noise
# isn't mistaken for a mis-specified network.
TOPO_CONFIGS = {
    "er":      ("er",   3, None, 3.0, "Erdos-Renyi, k=3 (reference baseline, rerun at 8 seeds)"),
    "ring_k2": ("ring", 2, None, 2.0, "k-regular ring lattice, k=2 (exact degree 2)"),
    "ring_k4": ("ring", 4, None, 4.0, "k-regular ring lattice, k=4 (exact degree 4)"),
    "ba_m1":   ("ba",   2, 1,    2.0, "Barabasi-Albert, m=1 explicit (mean degree ~2)"),
    "ba_m2":   ("ba",   4, 2,    4.0, "Barabasi-Albert, m=2 explicit (mean degree ~4)"),
}

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "output", "experiment_topology")
SUMMARY_CSV = os.path.join(OUTDIR, "topology_robustness_summary.csv")


def sanity_check(label, topology, k, m, intended_degree, N=50, seed=42):
    """Build one network + run one mu=lam=1 cell; report degree stats and (h,s).

    Fails LOUDLY (raises AssertionError) if the realised mean degree is not
    within tolerance of `intended_degree` -- this is the guard that would
    have caught both the ba1/ba2-collision bug and the odd-k ring-degree bug.
    ER uses a wider tolerance (DEGREE_TOL_ER) than ring/BA (DEGREE_TOL) since
    it is a random graph with genuine sampling variance in realised degree.
    """
    from dataclasses import replace
    from params import BASELINE
    from model.simulation import run
    from analysis.metrics import steady_state
    import networkx as nx

    G, W = build_network(N, k, topology, seed, m=m)
    degs = [d for _, d in G.degree()]
    mean_deg = float(np.mean(degs))
    connected = nx.is_connected(G)
    print(f"\n[sanity] {label}: topology={topology!r} k={k} m={m} N={N}")
    print(f"  realised mean degree = {mean_deg:.3f}  (min={min(degs)}, max={max(degs)})")
    print(f"  connected            = {connected}")

    tol = DEGREE_TOL_ER if topology == "er" else DEGREE_TOL
    if abs(mean_deg - intended_degree) > tol:
        raise AssertionError(
            f"[{label}] realised mean degree {mean_deg:.3f} is more than "
            f"+/-{tol} away from the intended target {intended_degree:.3f} "
            f"(topology={topology!r}, k={k}, m={m}). Refusing to run the sweep "
            f"on a mis-specified network -- fix the topology/k/m before rerunning."
        )

    # For the ABM run itself, BA's m is derived inside model.simulation from
    # Params.k via max(1, k // 2); the (k, m) pairs in TOPO_CONFIGS are chosen
    # so that derivation reproduces the same explicit m used in this sanity
    # check (k=2 -> m=1, k=4 -> m=2), so no change to Params/simulation.run is
    # needed for the full sweep to use the correct m.
    p = replace(BASELINE, T=T_DEFAULT, seed=seed, topology=topology, k=k,
                mu=1.0, lam=1.0, nu=1.0, phi=PHI, delta_loc=DELTA_LOC, delta_glob=DELTA_GLOB)
    st = steady_state(run(p))
    print(f"  mu=lam=1 cell: h_ss={st['h_ss']:.3f}  s_ss={st['s_ss']:.3f}  "
          f"(plausible range: both in [0,1], not NaN)")
    assert 0.0 <= st["h_ss"] <= 1.0 and 0.0 <= st["s_ss"] <= 1.0, "sanity check failed: h/s out of range"
    return mean_deg


def boundary_vs_lambda(res):
    """
    For each lambda row, find the mu at which R = s - h crosses zero
    (descending, i.e. green-club side -> race-to-bottom side as mu increases).
    Returns (lam_values, mu_star) with NaN where no crossing exists in range.
    """
    xs, ys = res["xvals"], res["yvals"]   # xs = mu grid, ys = lam grid
    R = mean_R(res)
    mu_star = np.full(len(ys), np.nan)
    for iy in range(len(ys)):
        mu_star[iy] = crossing(xs, R[iy, :], target=0.0, log=True)
    return ys, mu_star


def summarize_boundary(lam_vals, mu_star):
    ratio = mu_star / lam_vals
    valid = ~np.isnan(ratio)
    if valid.sum() == 0:
        return dict(median_ratio=np.nan, mean_ratio=np.nan, n_rows=0)
    return dict(median_ratio=float(np.median(ratio[valid])),
                mean_ratio=float(np.mean(ratio[valid])),
                n_rows=int(valid.sum()))


def run_topology(label, n=N_GRID_DEFAULT, seeds=SEEDS_DEFAULT, T=T_DEFAULT):
    topology, k, m, intended_degree, desc = TOPO_CONFIGS[label]
    print(f"\n{'='*70}\nTOPOLOGY: {label}  ({desc})\n{'='*70}")

    mean_deg = sanity_check(label, topology, k, m, intended_degree)

    fixed = dict(phi=PHI, delta_loc=DELTA_LOC, delta_glob=DELTA_GLOB, nu=None,
                 topology=topology, k=k)
    res = run_grid("mu",  np.exp(np.linspace(np.log(0.02), np.log(20.0), n)),
                   "lam", np.exp(np.linspace(np.log(0.1),  np.log(5.0),  n)),
                   fixed, seeds=seeds, T=T)

    name = f"phase_speed_{label}"
    _save_csv_with_topology(res, name, topology, k, m)
    png_path = os.path.join(OUTDIR, f"phase_speed_{label}.png")
    plot_speed_map(res, png_path,
                   f"Regulatory outcome by firm mobility $\\mu$ and institutional speed "
                   f"$\\lambda$ — {desc}")

    lam_vals, mu_star = boundary_vs_lambda(res)
    stats = summarize_boundary(lam_vals, mu_star)
    reg = modal_regime(res)
    green_share = float(np.mean(reg == 2))
    rtb_share = float(np.mean(reg == 0))

    print(f"\n  boundary mu*(lambda) [R=0 crossing], median mu*/lambda = "
          f"{stats['median_ratio']:.3f}  (n_rows_with_crossing={stats['n_rows']}/{n})")
    print(f"  green-club share of cells = {green_share:.3f}   race share = {rtb_share:.3f}")

    _append_summary(label, topology, k, m, intended_degree, mean_deg, seeds, stats, green_share, rtb_share)
    _save_boundary_csv(label, lam_vals, mu_star)

    return dict(label=label, mean_deg=mean_deg, stats=stats,
                green_share=green_share, rtb_share=rtb_share)


def _save_csv_with_topology(res, name, topology, k, m, outdir=OUTDIR):
    """Same schema as sweep_lib.save_csv but tags rows with topology/k/m for traceability."""
    os.makedirs(outdir, exist_ok=True)
    xs, ys = res["xvals"], res["yvals"]
    H, S, ns = res["H"], res["S"], res["seeds"]
    R, reg, pg = mean_R(res), modal_regime(res), p_green(res)
    path = os.path.join(outdir, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([res["xname"], res["yname"], "regime", "R", "h_mean", "s_mean",
                    "p_green", "topology", "k", "m"])
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                w.writerow([f"{xv:.6g}", f"{yv:.6g}", REGIME_NAMES[reg[iy, ix]],
                            f"{R[iy, ix]:.4f}", f"{np.nanmean(H[iy, ix]):.4f}",
                            f"{np.nanmean(S[iy, ix]):.4f}", f"{pg[iy, ix]:.3f}",
                            topology, k, m if m is not None else ""])
    print(f"  saved {path}")


def _save_boundary_csv(label, lam_vals, mu_star, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"phase_speed_{label}_boundary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam", "mu_star", "mu_star_over_lam"])
        for lam, mu in zip(lam_vals, mu_star):
            ratio = mu / lam if not np.isnan(mu) else ""
            w.writerow([f"{lam:.5g}", f"{mu:.5g}" if not np.isnan(mu) else "", ratio])
    print(f"  saved {path}")


def _append_summary(label, topology, k, m, intended_degree, mean_deg, seeds, stats,
                     green_share, rtb_share, path=SUMMARY_CSV):
    os.makedirs(OUTDIR, exist_ok=True)
    header = ["label", "topology", "k_or_m", "intended_degree", "realised_mean_degree",
              "seeds", "boundary_median_mu_over_lam", "boundary_mean_mu_over_lam",
              "n_rows_with_crossing", "green_club_share", "race_to_bottom_share"]
    k_or_m = m if (topology == "ba" and m is not None) else k
    row = [label, topology, k_or_m, f"{intended_degree:.3f}", f"{mean_deg:.3f}", seeds,
           f"{stats['median_ratio']:.4f}" if not np.isnan(stats['median_ratio']) else "",
           f"{stats['mean_ratio']:.4f}" if not np.isnan(stats['mean_ratio']) else "",
           stats["n_rows"], f"{green_share:.4f}", f"{rtb_share:.4f}"]
    write_header = not os.path.exists(path)
    if not write_header:
        # avoid duplicate rows for the same label across reruns
        with open(path) as f:
            existing = list(csv.reader(f))
        existing = [r for r in existing if r and r[0] != label]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(existing)
        write_header = len(existing) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"  appended summary row to {path}")


def rebuild_summary_from_scratch():
    """Delete any existing summary CSV so _append_summary starts a fresh file
    with exactly the rows produced by this run (used by `all`)."""
    if os.path.exists(SUMMARY_CSV):
        os.remove(SUMMARY_CSV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["er", "ring", "ring_k2", "ring_k4",
                                       "ba", "ba_m1", "ba_m2", "all"])
    ap.add_argument("--n", type=int, default=N_GRID_DEFAULT)
    ap.add_argument("--seeds", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    args = ap.parse_args()

    labels = {
        "er":      ["er"],
        "ring":    ["ring_k2", "ring_k4"],
        "ring_k2": ["ring_k2"],
        "ring_k4": ["ring_k4"],
        "ba":      ["ba_m1", "ba_m2"],
        "ba_m1":   ["ba_m1"],
        "ba_m2":   ["ba_m2"],
        "all":     ["er", "ring_k2", "ring_k4", "ba_m1", "ba_m2"],
    }[args.which]

    if args.which == "all":
        rebuild_summary_from_scratch()

    results = []
    for label in labels:
        results.append(run_topology(label, n=args.n, seeds=args.seeds, T=args.T))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        ratio = r["stats"]["median_ratio"]
        side = "AT mu=lam" if np.isnan(ratio) else ("RIGHT of mu=lam (RTB-favouring)" if ratio > 1
              else "LEFT of mu=lam (green-club-favouring)" if ratio < 1 else "ON mu=lam")
        print(f"  {r['label']:8s} mean_deg={r['mean_deg']:.2f}  "
              f"median mu*/lam={ratio:.3f}  [{side}]  "
              f"green={r['green_share']:.2f} rtb={r['rtb_share']:.2f}")

    if args.which == "all":
        _verify_all(results)


def _verify_all(results):
    """Print the checks requested for the full 'all' run."""
    print(f"\n{'='*70}\nVERIFICATION\n{'='*70}")

    by_label = {r["label"]: r for r in results}

    # 1. ba_m1 and ba_m2 boundary files no longer identical.
    p1 = os.path.join(OUTDIR, "phase_speed_ba_m1_boundary.csv")
    p2 = os.path.join(OUTDIR, "phase_speed_ba_m2_boundary.csv")
    if os.path.exists(p1) and os.path.exists(p2):
        with open(p1) as f1, open(p2) as f2:
            identical = f1.read() == f2.read()
        print(f"  ba_m1 vs ba_m2 boundary CSVs identical? {identical}  "
              f"(expected: False)")
    else:
        print("  [skip] ba_m1/ba_m2 boundary files not both present in this run")

    # 2. realised mean degrees close to target.
    for label, target in [("ring_k2", 2.0), ("ring_k4", 4.0),
                           ("ba_m1", 2.0), ("ba_m2", 4.0)]:
        if label in by_label:
            print(f"  {label}: realised mean degree = {by_label[label]['mean_deg']:.3f} "
                  f"(target ~{target})")

    # 3. summary table row count.
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV) as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
        print(f"  summary table rows = {n_rows}  (expected: 5 -> er, ring_k2, ring_k4, ba_m1, ba_m2)")

    # 4. boundary side per topology.
    for r in results:
        ratio = r["stats"]["median_ratio"]
        if np.isnan(ratio):
            side = "no crossing found"
        elif ratio < 0.9:
            side = "< 1 (left of mu=lam, green-club-favouring)"
        elif ratio > 1.1:
            side = "> 1 (right of mu=lam, RTB-favouring)"
        else:
            side = "~= 1 (on mu=lam)"
        print(f"  {r['label']:8s} median mu*/lambda = {ratio:.3f}  -> {side}")


if __name__ == "__main__":
    main()
