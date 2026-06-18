"""
Read the CSVs produced by run_model.py and generate all diagnostic plots.
Run:  python plot_results.py
All figures are saved to output/figures/.

Figure inventory (8 figures, no duplicates):
  1  aggregate_dynamics    h(t) and s(t)
  2  phase_portrait        trajectory in (h, s) space
  3  profits               π̄_H/L levels  +  profit gap π̄_H − π̄_L
  4  prices_by_policy      mean ± IQR price by policy type  +  strict−lax gap
  5  firm_composition      mean H/L firm counts per policy type (stacked area)
  6  leakage_dynamics      H-firm density per policy type  +  relocation dynamics
  7  welfare_components    welfare decomposition per policy type
  8  trade_policy          strict/lax price gap  |  b_SL  |  import competition
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

agg = pd.read_csv("output/aggregate.csv")
jur = pd.read_csv("output/jurisdiction.csv")
rel = pd.read_csv("output/relocations.csv")

os.makedirs("output/figures", exist_ok=True)

PERIODS = agg["period"].values
POLICY_COLOR = {"S": "#2166ac", "L": "#d6604d"}


def save(fig, name):
    path = f"output/figures/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. h(t) and s(t) — aggregate dynamics
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(12, 4))

ax_h.plot(PERIODS, agg["h"], color="#d6604d", linewidth=2)
ax_h.set_xlabel("Period")
ax_h.set_ylabel("Share")
ax_h.set_ylim(-0.05, 1.05)
ax_h.set_title("h(t) — high-emission firm share")
ax_h.grid(alpha=0.3)

ax_s.plot(PERIODS, agg["s"], color="#2166ac", linewidth=2)
ax_s.set_xlabel("Period")
ax_s.set_ylabel("Share")
ax_s.set_ylim(-0.05, 1.05)
ax_s.set_title("s(t) — strict jurisdiction share")
ax_s.grid(alpha=0.3)

fig.suptitle("Aggregate dynamics", fontsize=13)
fig.tight_layout()
save(fig, "1_aggregate_dynamics")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Phase portrait — trajectory in (h, s) space
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 5))
h, s = agg["h"].values, agg["s"].values
ax.plot(h, s, color="black", linewidth=1, zorder=1)
ax.scatter(h[0],  s[0],  color="green", s=80, zorder=5, label="start")
ax.scatter(h[-1], s[-1], color="red",   s=80, zorder=5, label="end")
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
# 3. Profits — levels and profit gap
#
# Left:  π̄_H and π̄_L over time (drives emission replicator direction)
# Right: π̄_H − π̄_L (the quantity that enters ḣ = h(1−h)(π̄_H − π̄_L);
#         above zero → H expands, below → H shrinks)
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_lev, ax_gap) = plt.subplots(1, 2, figsize=(13, 4))

ax_lev.plot(PERIODS, agg["pi_H_bar"], color="#d6604d", linewidth=2, label="π̄_H  (high-emission)")
ax_lev.plot(PERIODS, agg["pi_L_bar"], color="#4dac26", linewidth=2, label="π̄_L  (low-emission)")
ax_lev.set_xlabel("Period")
ax_lev.set_ylabel("Average net profit")
ax_lev.set_title("Profit levels")
ax_lev.legend()
ax_lev.grid(alpha=0.3)

gap_pi = agg["pi_H_bar"] - agg["pi_L_bar"]
ax_gap.plot(PERIODS, gap_pi, color="black", linewidth=2)
ax_gap.axhline(0, color="grey", linewidth=1, linestyle="--")
ax_gap.fill_between(PERIODS, gap_pi, 0,
                    where=(gap_pi > 0), color="#d6604d", alpha=0.2,
                    label="H-type advantaged  (h grows)")
ax_gap.fill_between(PERIODS, gap_pi, 0,
                    where=(gap_pi < 0), color="#4dac26", alpha=0.2,
                    label="L-type advantaged  (h shrinks)")
ax_gap.set_xlabel("Period")
ax_gap.set_ylabel("π̄_H − π̄_L")
ax_gap.set_title("Profit gap  (replicator signal for h)")
ax_gap.legend(fontsize=8)
ax_gap.grid(alpha=0.3)

fig.suptitle("Firm profit dynamics", fontsize=13)
fig.tight_layout()
save(fig, "3_profits")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prices by policy type — mean ± IQR + strict/lax gap
#
# Left:  mean price per policy type with shaded 25–75th percentile band.
#        More informative than 50 overlapping lines.
# Right: mean(p*_strict) − mean(p*_lax) over time.
#        Sign tells you which type has higher equilibrium prices.
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_lev, ax_gap) = plt.subplots(1, 2, figsize=(13, 4))

for pol_code, pol_label in [("S", "Strict"), ("L", "Lax")]:
    sub = (jur[jur["policy"] == pol_code]
           .groupby("period")["price"]
           .agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
           .reset_index())
    if sub.empty:
        continue
    color = POLICY_COLOR[pol_code]
    ax_lev.plot(sub["period"], sub["mean"], color=color, linewidth=2, label=pol_label)
    ax_lev.fill_between(sub["period"], sub["q25"], sub["q75"], color=color, alpha=0.15)

ax_lev.set_xlabel("Period")
ax_lev.set_ylabel("Equilibrium price p*")
ax_lev.set_title("Mean price ± IQR by policy type")
ax_lev.legend()
ax_lev.grid(alpha=0.3)

strict_p = jur[jur["policy"] == "S"].groupby("period")["price"].mean()
lax_p    = jur[jur["policy"] == "L"].groupby("period")["price"].mean()
gap_p    = (strict_p - lax_p).reindex(PERIODS)
ax_gap.plot(PERIODS, gap_p, color="black", linewidth=2)
ax_gap.axhline(0, linestyle="--", color="grey", linewidth=1)
ax_gap.fill_between(PERIODS, gap_p, 0,
                    where=(gap_p > 0), color="#d6604d", alpha=0.15,
                    label="Strict more expensive")
ax_gap.fill_between(PERIODS, gap_p, 0,
                    where=(gap_p < 0), color="#2166ac", alpha=0.15,
                    label="Lax more expensive")
ax_gap.set_xlabel("Period")
ax_gap.set_ylabel("p*_strict − p*_lax")
ax_gap.set_title("Strict / lax price gap")
ax_gap.legend(fontsize=8)
ax_gap.grid(alpha=0.3)

fig.suptitle("Equilibrium prices", fontsize=13)
fig.tight_layout()
save(fig, "4_prices_by_policy")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Firm composition — mean H/L counts, strict vs lax
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, pol_label, pol_code in zip(axes, ["Strict jurisdictions", "Lax jurisdictions"], ["S", "L"]):
    sub = (jur[jur["policy"] == pol_code]
           .groupby("period")[["f_H", "f_L"]]
           .mean()
           .reset_index())
    if sub.empty:
        ax.text(0.5, 0.5, f"No {pol_label.lower()}\nin this run",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title(pol_label)
        continue
    ax.stackplot(sub["period"], sub["f_H"], sub["f_L"],
                 labels=["Mean H-firms", "Mean L-firms"],
                 colors=["#d6604d", "#4dac26"], alpha=0.85)
    ax.set_title(pol_label, color=POLICY_COLOR[pol_code], fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean firm count per jurisdiction")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

fig.suptitle("Mean firm composition by policy type", fontsize=13)
fig.tight_layout()
save(fig, "5_firm_composition")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Carbon leakage dynamics
#
# Left:  Mean H-firm count per jurisdiction, split by current policy.
#        Divergence between the two lines = leakage in action.
# Right: Relocation events per period + fraction that are leakage moves (S→L).
#        Requires p.relocate = True to show anything.
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_dens, ax_rel) = plt.subplots(1, 2, figsize=(13, 4))

# Left — H-firm density per policy type
for pol_code, pol_label in [("S", "Strict"), ("L", "Lax")]:
    sub = (jur[jur["policy"] == pol_code]
           .groupby("period")["f_H"]
           .mean()
           .reset_index())
    if not sub.empty:
        ax_dens.plot(sub["period"], sub["f_H"],
                     color=POLICY_COLOR[pol_code], linewidth=2, label=pol_label)

ax_dens.set_xlabel("Period")
ax_dens.set_ylabel("Mean H-firm count per jurisdiction")
ax_dens.set_title("H-firm density by policy type\n(divergence = leakage)")
ax_dens.legend()
ax_dens.grid(alpha=0.3)

# Right — relocation count + leakage fraction
ax_rel2 = ax_rel.twinx()

reloc_per_period = agg[["period", "n_relocations"]].copy()
ax_rel.bar(reloc_per_period["period"], reloc_per_period["n_relocations"],
           color="steelblue", alpha=0.4, label="Total relocations", width=1.0)

if len(rel) > 0:
    leakage_frac = (rel.groupby("period")["is_leakage"].mean()
                    .reindex(PERIODS, fill_value=np.nan))
    ax_rel2.plot(PERIODS, leakage_frac, color="#d6604d", linewidth=1.5,
                 label="Leakage fraction (S→L)")
    ax_rel2.set_ylim(-0.05, 1.05)
    ax_rel2.set_ylabel("Fraction of moves that are leakage (S→L)", color="#d6604d")
    ax_rel2.tick_params(axis="y", labelcolor="#d6604d")
else:
    ax_rel.text(0.5, 0.5, "No relocations\n(p.relocate = False)",
                ha="center", va="center", transform=ax_rel.transAxes,
                fontsize=11, color="grey")

ax_rel.set_xlabel("Period")
ax_rel.set_ylabel("Relocations per period")
ax_rel.set_title("Relocation events + leakage fraction")

lines1, labels1 = ax_rel.get_legend_handles_labels()
lines2, labels2 = ax_rel2.get_legend_handles_labels()
ax_rel.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax_rel.grid(alpha=0.2)

fig.suptitle("Carbon leakage dynamics", fontsize=13)
fig.tight_layout()
save(fig, "6_leakage_dynamics")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Welfare components — strict vs lax, stacked decomposition
#
# Each panel: consumer surplus (green) stacked with fiscal revenue (yellow),
# minus damage (red, shown as the portion subtracted below the surplus stack).
# Black dashed line = total welfare per capita.
#
# Note: damage_pc is stored as a positive number; it enters welfare negatively.
# ─────────────────────────────────────────────────────────────────────────────

jur["TR_pc"] = jur["TR"] / jur["population"].clip(lower=1.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, pol_label, pol_code in zip(axes, ["Strict jurisdictions", "Lax jurisdictions"], ["S", "L"]):
    sub = (jur[jur["policy"] == pol_code]
           .groupby("period")[["cs_pc", "TR_pc", "damage_pc", "welfare_pc"]]
           .mean()
           .reset_index())

    if sub.empty:
        ax.text(0.5, 0.5, f"No {pol_label.lower()}\nin this run",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title(pol_label)
        continue

    t   = sub["period"].values
    cs  = sub["cs_pc"].values
    tr  = sub["TR_pc"].values
    dmg = sub["damage_pc"].values   # positive number; subtracted from welfare

    # Stack: [0 → cs] green, [cs → cs+tr] yellow
    ax.fill_between(t, 0,      cs,      color="#91cf60", alpha=0.85, label="Consumer surplus")
    ax.fill_between(t, cs,     cs + tr, color="#fee08b", alpha=0.85, label="Fiscal revenue / P")
    # Damage is carved out below the welfare line (shown as subtracted area)
    ax.fill_between(t, cs + tr - dmg, cs + tr,
                    color="#d73027", alpha=0.70, label="Env. damage")

    ax.plot(t, sub["welfare_pc"].values, color="black",
            linewidth=2, linestyle="--", label="Total W/P")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(pol_label, color=POLICY_COLOR[pol_code], fontweight="bold")
    ax.set_xlabel("Period")
    ax.set_ylabel("Mean per-capita welfare")
    ax.grid(alpha=0.2)

handles = [
    mpatches.Patch(color="#91cf60", label="Consumer surplus"),
    mpatches.Patch(color="#fee08b", label="Fiscal revenue / P"),
    mpatches.Patch(color="#d73027", label="Env. damage"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Total W/P"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Mean welfare components by policy type", fontsize=13)
fig.tight_layout()
save(fig, "7_welfare_components")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Trade / policy mechanisms
#
# Left:   Strict/lax price gap — shows market effect of carbon pricing.
# Centre: b_SL over time — network correction to payoff matrix (eq. 3.51);
#         amplifies or dampens the strict-vs-lax policy advantage.
# Right:  Mean imported quantity per market, split by policy type.
#         Shows whether strict or lax markets attract more foreign competition.
# ─────────────────────────────────────────────────────────────────────────────

import params as cfg

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Left — price gap
gap_p_full = (strict_p - lax_p).reindex(PERIODS)
axes[0].plot(PERIODS, gap_p_full, color="black", linewidth=2)
axes[0].axhline(0, linestyle="--", color="grey", linewidth=1)
axes[0].fill_between(PERIODS, gap_p_full, 0,
                     where=(gap_p_full > 0), color="#d6604d", alpha=0.15,
                     label="Strict more expensive")
axes[0].fill_between(PERIODS, gap_p_full, 0,
                     where=(gap_p_full < 0), color="#2166ac", alpha=0.15,
                     label="Lax more expensive")
axes[0].set_xlabel("Period")
axes[0].set_ylabel("p*_strict − p*_lax")
axes[0].set_title("Price gap: strict vs lax")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Centre — b_SL network correction
axes[1].plot(PERIODS, agg["b_SL"], color="purple", linewidth=2)
axes[1].axhline(0, linestyle="--", color="grey", linewidth=1)
axes[1].fill_between(PERIODS, agg["b_SL"], 0,
                     where=(agg["b_SL"] > 0), color="purple", alpha=0.15,
                     label="Amplifies S adoption")
axes[1].fill_between(PERIODS, agg["b_SL"], 0,
                     where=(agg["b_SL"] < 0), color="orange", alpha=0.25,
                     label="Dampens S adoption")
axes[1].set_xlabel("Period")
axes[1].set_ylabel("b_SL")
axes[1].set_title("Network correction b_SL  (eq. 3.51)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Right — import competition by policy type
for pol_code, color, label in [("S", POLICY_COLOR["S"], "Strict markets"),
                                ("L", POLICY_COLOR["L"], "Lax markets")]:
    sub = jur[jur["policy"] == pol_code].groupby("period")["f_imported"].mean()
    if not sub.empty:
        axes[2].plot(sub.index, sub.values, color=color, linewidth=2, label=label)
overall = jur.groupby("period")["f_imported"].mean()
axes[2].plot(overall.index, overall.values, color="black",
             linewidth=1.5, linestyle="--", label="Economy-wide mean")
axes[2].set_xlabel("Period")
axes[2].set_ylabel("Mean imported quantity per market")
axes[2].set_title("Import competition by market type")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

fig.suptitle(f"Trade / policy mechanisms   (g={cfg.g},  τ={cfg.tau},  τ_BA={cfg.tau_BA},  t={cfg.t})",
             fontsize=11)
fig.tight_layout()
save(fig, "8_trade_policy")


# ─────────────────────────────────────────────────────────────────────────────

print(f"\nAll figures saved to output/figures/")
