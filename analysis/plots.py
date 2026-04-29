"""
Plotting utilities for simulation output.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(results: dict, title: str = "") -> plt.Figure:
    """Plot h(t) and s(t) on the same axes."""
    fig, ax = plt.subplots()
    ax.plot(results["t"], results["h"], label="h  (high-emission share)", color="tab:red")
    ax.plot(results["t"], results["s"], label="s  (strict jurisdiction share)", color="tab:blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_phase_portrait(results: dict, title: str = "") -> plt.Figure:
    """Trajectory in (h, s) phase space."""
    fig, ax = plt.subplots()
    ax.plot(results["h"], results["s"], color="black", linewidth=0.8)
    ax.scatter(results["h"][0],  results["s"][0],  color="green",  zorder=5, label="start")
    ax.scatter(results["h"][-1], results["s"][-1], color="red",    zorder=5, label="end")
    ax.set_xlabel("h  (high-emission firm share)")
    ax.set_ylabel("s  (strict jurisdiction share)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_firm_distribution(results: dict, step: int = -1, title: str = "") -> plt.Figure:
    """Bar chart of high- vs low-emission firms per jurisdiction at a given step."""
    f_H = results["f_H"][step]
    f_L = results["f_L"][step]
    N = len(f_H)
    x = np.arange(N)
    fig, ax = plt.subplots(figsize=(max(8, N // 3), 4))
    ax.bar(x, f_H, label="High-emission", color="tab:red",  alpha=0.8)
    ax.bar(x, f_L, bottom=f_H, label="Low-emission", color="tab:green", alpha=0.8)
    ax.set_xlabel("Jurisdiction")
    ax.set_ylabel("Number of firms")
    ax.legend()
    ax.set_title(title or f"Firm distribution at step {step}")
    fig.tight_layout()
    return fig


def plot_heatmap(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_matrix: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
) -> plt.Figure:
    """Generic heatmap for parameter sweeps (e.g. µ/λ vs δ → outcome)."""
    fig, ax = plt.subplots()
    im = ax.imshow(
        z_matrix,
        origin="lower",
        aspect="auto",
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig
