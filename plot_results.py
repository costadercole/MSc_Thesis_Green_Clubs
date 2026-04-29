"""
Read the CSVs produced by run_model.py and generate all diagnostic plots.
Run:  python plot_results.py
All figures are saved to output/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

agg  = pd.read_csv("output/aggregate.csv")
jur  = pd.read_csv("output/jurisdiction.csv")
rel  = pd.read_csv("output/relocations.csv")
pol  = pd.read_csv("output/policy_changes.csv")

os.makedirs("output/figures", exist_ok=True)

JURISDICTIONS = sorted(jur["jurisdiction"].unique())
PERIODS       = agg["period"].values

# Colour helpers
POLICY_COLOR = {"S": "#2166ac", "L": "#d6604d"}   # blue=strict, red=lax

def jur_colors(df_row_series):
    """Return a colour per period based on policy at that period."""
    return [POLICY_COLOR[p] for p in df_row_series]

def save(fig, name):
    path = f"output/figures/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. h(t) and s(t) — aggregate dynamics
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(PERIODS, agg["h"], color="#d6604d", linewidth=2, label="h  (high-emission firm share)")
ax.plot(PERIODS, agg["s"], color="#2166ac", linewidth=2, label="s  (strict jurisdiction share)")
ax.set_xlabel("Period")
ax.set_ylabel("Share")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Aggregate dynamics: firm type and policy shares")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "1_aggregate_dynamics")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Phase portrait — trajectory in (h, s) space
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 5))
h, s = agg["h"].values, agg["s"].values
ax.plot(h, s, color="black", linewidth=1, zorder=1)
ax.scatter(h[0],  s[0],  color="green", s=80, zorder=5, label="start")
ax.scatter(h[-1], s[-1], color="red",   s=80, zorder=5, label="end")
# arrow showing direction
for k in range(0, len(h) - 1, max(1, len(h) // 8)):
    ax.annotate("", xy=(h[k+1], s[k+1]), xytext=(h[k], s[k]),
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
ax.set_xlabel("h  (high-emission firm share)")
ax.set_ylabel("s  (strict jurisdiction share)")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_title("Phase portrait")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "2_phase_portrait")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Average profits over time — π̄_H vs π̄_L
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(PERIODS, agg["pi_H_bar"], color="#d6604d", linewidth=2, label="π̄_H  (high-emission)")
ax.plot(PERIODS, agg["pi_L_bar"], color="#4dac26", linewidth=2, label="π̄_L  (low-emission)")
ax.set_xlabel("Period")
ax.set_ylabel("Average net profit")
ax.set_title("Economy-wide average profits")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "3_average_profits")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Equilibrium price per jurisdiction — coloured by policy
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
for jname in JURISDICTIONS:
    sub = jur[jur["jurisdiction"] == jname].sort_values("period")
    # draw segments coloured by policy
    for k in range(len(sub) - 1):
        row = sub.iloc[k]
        ax.plot(
            [row["period"], sub.iloc[k+1]["period"]],
            [row["price"],  sub.iloc[k+1]["price"]],
            color=POLICY_COLOR[row["policy"]],
            linewidth=2,
        )
    # label at last point
    last = sub.iloc[-1]
    ax.text(last["period"] + 0.1, last["price"], jname, fontsize=8, va="center")

legend_handles = [
    mpatches.Patch(color=POLICY_COLOR["S"], label="Strict policy"),
    mpatches.Patch(color=POLICY_COLOR["L"], label="Lax policy"),
]
ax.legend(handles=legend_handles)
ax.set_xlabel("Period")
ax.set_ylabel("Equilibrium price p*")
ax.set_title("Equilibrium prices by jurisdiction (colour = current policy)")
ax.grid(alpha=0.3)
save(fig, "4_prices_by_jurisdiction")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Firm composition — mean H and L firm counts, split by policy type
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, pol_label, pol_code in zip(axes, ["Strict jurisdictions", "Lax jurisdictions"], ["S", "L"]):
    sub = jur[jur["policy"] == pol_code].groupby("period")[["f_H", "f_L"]].mean().reset_index()
    ax.stackplot(
        sub["period"],
        sub["f_H"], sub["f_L"],
        labels=["Mean H-firms", "Mean L-firms"],
        colors=["#d6604d", "#4dac26"],
        alpha=0.85,
    )
    ax.set_title(pol_label, color=POLICY_COLOR[pol_code], fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean number of firms per jurisdiction")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

fig.suptitle("Mean firm composition by policy type")
fig.tight_layout()
save(fig, "5_firm_composition")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Welfare components — mean per policy type, stacked area over time
# ─────────────────────────────────────────────────────────────────────────────

components  = ["wage_pc", "cs_pc", "tax_pc", "tariff_pc", "damage_pc"]
comp_labels = ["Wage", "Consumer surplus", "Tax revenue", "Tariff", "Env. damage"]
comp_colors = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, pol_label, pol_code in zip(axes, ["Strict jurisdictions", "Lax jurisdictions"], ["S", "L"]):
    sub = (jur[jur["policy"] == pol_code]
           .groupby("period")[components + ["welfare_pc"]]
           .mean()
           .reset_index())

    if sub.empty:
        ax.text(0.5, 0.5, f"No {pol_label.lower()}\nin this run",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title(pol_label)
        continue

    bottoms_pos = np.zeros(len(sub))
    bottoms_neg = np.zeros(len(sub))
    for comp, label, color in zip(components, comp_labels, comp_colors):
        vals = sub[comp].values
        pos  = np.where(vals >= 0, vals, 0)
        neg  = np.where(vals <  0, vals, 0)
        ax.stackplot(sub["period"], pos,
                     baseline="zero", colors=[color], alpha=0.85, labels=[label])
        if neg.any():
            ax.fill_between(sub["period"], bottoms_neg, bottoms_neg + neg,
                            color=color, alpha=0.85)
            bottoms_neg += neg

    ax.plot(sub["period"], sub["welfare_pc"], color="black",
            linewidth=2, linestyle="--", label="Total W/P")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(pol_label, color=POLICY_COLOR[pol_code], fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean per-capita welfare")
    ax.grid(alpha=0.2)

handles = [mpatches.Patch(color=c, label=l) for c, l in zip(comp_colors, comp_labels)]
handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Total W/P"))
fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Mean welfare components by policy type")
fig.tight_layout()
save(fig, "6_welfare_components")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Import competition — mean effective importers by policy type + overall
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
for pol_code, pol_label, color in [("S", "Strict", POLICY_COLOR["S"]),
                                    ("L", "Lax",    POLICY_COLOR["L"])]:
    sub = (jur[jur["policy"] == pol_code]
           .groupby("period")["f_imported"].mean().reset_index())
    if not sub.empty:
        ax.plot(sub["period"], sub["f_imported"], color=color,
                linewidth=2, label=f"Mean — {pol_label}")

overall = jur.groupby("period")["f_imported"].mean().reset_index()
ax.plot(overall["period"], overall["f_imported"], color="black",
        linewidth=1.5, linestyle="--", label="Economy-wide mean")

ax.set_xlabel("Period")
ax.set_ylabel("Effective imported competitors")
ax.set_title("Import competition by policy type")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "7_import_competition")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Relocation events scatter
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
if len(rel) > 0:
    colors = ["#2166ac" if d > 0 else "#d6604d" for d in rel["delta_pi"]]
    ax.scatter(rel["period"], rel["delta_pi"], c=colors, s=60, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    legend_handles = [
        mpatches.Patch(color="#2166ac", label="Moved to higher-profit jurisdiction"),
        mpatches.Patch(color="#d6604d", label="Moved to lower-profit jurisdiction"),
    ]
    ax.legend(handles=legend_handles)
else:
    ax.text(0.5, 0.5, "No relocations occurred", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="grey")
ax.set_xlabel("Period")
ax.set_ylabel("Δπ  (destination − origin profit)")
ax.set_title("Firm relocation events")
ax.grid(alpha=0.3)
save(fig, "8_relocations")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Policy change timeline
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, max(3, n_jur * 0.8 + 1)))
y_pos = {j: i for i, j in enumerate(JURISDICTIONS)}

# background: policy over time per jurisdiction
for jname in JURISDICTIONS:
    sub = jur[jur["jurisdiction"] == jname].sort_values("period")
    for _, row in sub.iterrows():
        ax.barh(y_pos[jname], 1, left=row["period"] - 0.5,
                color=POLICY_COLOR[row["policy"]], alpha=0.25, height=0.6)

# overlay change events
if len(pol) > 0:
    for _, row in pol.iterrows():
        y = y_pos.get(row["jurisdiction"])
        if y is None:
            continue
        color = POLICY_COLOR[row["policy_after"]]
        ax.scatter(row["period"], y, color=color, s=120, zorder=5,
                   edgecolors="black", linewidths=0.7)
        ax.text(row["period"] + 0.1, y + 0.25,
                f"{row['policy_before']}→{row['policy_after']}",
                fontsize=7, va="bottom")

ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(list(y_pos.keys()))
ax.set_xlabel("Period")
ax.set_title("Policy timeline  (background = current policy, dot = switch event)")
legend_handles = [
    mpatches.Patch(color=POLICY_COLOR["S"], alpha=0.5, label="Strict"),
    mpatches.Patch(color=POLICY_COLOR["L"], alpha=0.5, label="Lax"),
]
ax.legend(handles=legend_handles, loc="upper right")
ax.grid(axis="x", alpha=0.3)
save(fig, "9_policy_timeline")


# ─────────────────────────────────────────────────────────────────────────────

print(f"\nAll figures saved to output/figures/")
