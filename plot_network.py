"""
Network visualization of the simulation.

Nodes  : jurisdictions
  colour  — green = strict (S), red = lax (L)
  size    — proportional to total firms in that jurisdiction
Edges  : neighbouring jurisdictions (from the network weight matrix W)

Usage:
  python plot_network.py                  # 5 snapshot panels, saved to output/figures/
  python plot_network.py --gif            # also save animated GIF (slow)
  python plot_network.py --period 500     # single-period figure
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as cfg
from model.network import build_network

# ── constants ────────────────────────────────────────────────────────────────
COLOR_STRICT = "#2ecc71"   # green
COLOR_LAX    = "#e74c3c"   # red
EDGE_COLOR   = "#aaaaaa"
EDGE_WIDTH   = 0.8
NODE_SCALE   = 30          # base multiplier: size = NODE_SCALE * f_total + NODE_MIN
NODE_MIN     = 40          # minimum node area so isolated nodes are visible

OUT_DIR      = "output/figures"


# ── helpers ──────────────────────────────────────────────────────────────────

def _layout(G: nx.Graph, topology: str) -> dict:
    """Fixed layout: circular for ring, spring otherwise."""
    if topology == "ring":
        return nx.circular_layout(G)
    return nx.spring_layout(G, seed=cfg.seed)


def _draw_period(ax, jur_period: pd.DataFrame, G: nx.Graph, pos: dict, period: int):
    """Draw one period's network state onto ax."""
    # Build per-node arrays aligned to node index
    nodes = sorted(G.nodes())
    n = len(nodes)

    colors = []
    sizes  = []
    for node in nodes:
        row = jur_period[jur_period["jurisdiction"] == f"J{node+1}"]
        if row.empty:
            colors.append(COLOR_LAX)
            sizes.append(NODE_MIN)
        else:
            policy  = row["policy"].values[0]
            f_total = int(row["f_total"].values[0])
            colors.append(COLOR_STRICT if policy == "S" else COLOR_LAX)
            sizes.append(NODE_SCALE * f_total + NODE_MIN)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=EDGE_COLOR, width=EDGE_WIDTH, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           nodelist=nodes, node_color=colors,
                           node_size=sizes, alpha=0.9)

    # Label only for small N
    if n <= 20:
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={v: str(v+1) for v in nodes},
                                font_size=6, font_color="white", font_weight="bold")

    s_frac = jur_period["policy"].eq("S").mean()
    ax.set_title(f"Period {period}\ns = {s_frac:.2f}", fontsize=9)
    ax.axis("off")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif",    action="store_true", help="save animated GIF")
    parser.add_argument("--period", type=int, default=None,
                        help="single period to plot (default: 5-panel snapshot)")
    args = parser.parse_args()

    # Load data
    jur_path = os.path.join("output", "jurisdiction.csv")
    if not os.path.exists(jur_path):
        sys.exit(f"Error: {jur_path} not found. Run run_model.py first.")
    jur = pd.read_csv(jur_path)
    T   = jur["period"].max()

    # Rebuild network
    G, W = build_network(cfg.N, cfg.k, cfg.topology, cfg.seed)
    pos  = _layout(G, cfg.topology)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── single-period figure ─────────────────────────────────────────────────
    if args.period is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        t = args.period
        _draw_period(ax, jur[jur["period"] == t], G, pos, t)
        _add_legend(fig)
        out = os.path.join(OUT_DIR, f"network_t{t:04d}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        return

    # ── 5-panel snapshot ─────────────────────────────────────────────────────
    snap_periods = _snapshot_periods(T, n=5)
    fig, axes = plt.subplots(1, len(snap_periods), figsize=(4 * len(snap_periods), 4.5))
    if len(snap_periods) == 1:
        axes = [axes]

    for ax, t in zip(axes, snap_periods):
        _draw_period(ax, jur[jur["period"] == t], G, pos, t)

    _add_legend(fig)
    fig.suptitle("Network dynamics  —  green = strict, red = lax\n"
                 "node size ∝ total firms", fontsize=10, y=1.01)
    out = os.path.join(OUT_DIR, "network_snapshots.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # ── animated GIF ─────────────────────────────────────────────────────────
    if args.gif:
        _save_gif(jur, G, pos, T)


def _snapshot_periods(T: int, n: int = 5) -> list[int]:
    """Pick n evenly-spaced periods including 1 and T."""
    if T <= n:
        return list(range(1, T + 1))
    step = (T - 1) / (n - 1)
    return [max(1, round(1 + i * step)) for i in range(n)]


def _add_legend(fig):
    patches = [
        mpatches.Patch(color=COLOR_STRICT, label="Strict (S)"),
        mpatches.Patch(color=COLOR_LAX,    label="Lax (L)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))


def _save_gif(jur, G, pos, T):
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Pillow not installed — skipping GIF. Run: pip install Pillow")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    periods = sorted(jur["period"].unique())

    def update(t):
        ax.clear()
        _draw_period(ax, jur[jur["period"] == t], G, pos, t)

    ani = FuncAnimation(fig, update, frames=periods, interval=80)
    out = os.path.join(OUT_DIR, "network_animation.gif")
    ani.save(out, writer=PillowWriter(fps=12))
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
