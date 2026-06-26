# Results summary — coevolutionary climate-club model

Coevolutionary agent-based model of firms and jurisdictions on a trade network
(Cournot goods market, Fermi imitation for jurisdiction policy, logistic-clipped
relocation for firms). All figures/CSVs in this folder. Numbers below are read
from the steady-state CSVs (mean over the final 20% of T=2000 runs, 4–6 seeds/cell)
and the welfare-decomposition layer (`analysis/welfare.py`, eq. 3.42).

---

## 1. Calibrated baseline

| Block | Parameters |
|---|---|
| Market | c_H=4, c_L=6 (Δc=2), carbon tax t=3, transport g=2.3, τ=0.5, τ_BA=0.5 |
| Welfare | δ_loc=1000, δ_glob=250, host benefit φ=1500 |
| Structural | a=20, b=1, N=50, M=500, k=3 (ER), σ_P=0, κ=1, κ_f=1e-3, ε=1e-4 |
| Contested anchor | (φ=1500, δ_loc=1000) sits on the structural boundary φ_crit≈1.5·δ_loc |

Achieved calibration moments: carbon leakage (competitive) 16.8%, import penetration
10.4%, price dispersion (CV) 10.7% — all inside their empirical target bands.

## 2. The economy (descriptive, mixed 50/50 snapshot)

| Quantity | Value |
|---|---|
| Firms per jurisdiction | 10 (500/50); ER mean degree ≈ 2.9 |
| Firm composition | ≈49% high-emission (dirty), ≈51% low-emission (clean) |
| Output split | **63% from dirty firms, 37% clean** (dirty produce more — lower cost) |
| Active firms per market | ≈3.1 active of ≈7.8 candidates |
| Mean price | 6.94 (strict 7.60 vs lax 6.29) |

**Trade is essentially all dirty.** High-emission firms export ≈19% of their output
(49% of them export at all); low-emission firms export ≈0% (priced out by transport
friction). So **≈100% of cross-border trade is emissions-intensive**, with overall
import penetration ≈12%. This single fact drives the instrument-equivalence result
in Experiment 2.

---

## 3. Experiment results

### Exp 1 — phase diagrams (regime by fundamentals and by speed)
- **Structural map (φ × δ_loc, neutral speed μ=λ):** clean diagonal boundary at
  **φ ≈ 1.5·δ_loc**. Regime set by the sign of the net hosting payoff (φ − δ_loc):
  damage-dominated → green club; jobs/tax-dominated → race to bottom.
- **Speed map (μ × λ, contested fundamentals):** boundary just **right of μ=λ** — the
  outcome is governed by the relative speed of capital vs institutions, with a
  critical ratio that drifts weakly with the absolute rates. Low μ/λ → green club
  (institutions outrun capital); high μ/λ → race to bottom (capital outruns
  institutions).

### Exp 2 — border carbon adjustment as a speed lever (CBAM)
- Critical border charge **τ\*(μ) rises with mobility**: ≈0.16 (slow capital) →
  ≈0.50 at μ/λ=1 → ≈0.63 at μ/λ=3 → saturating ≈0.65. A charge of only **~15–20% of
  the carbon tax** sustains the club even against capital several times faster than
  institutions.
- **The generic tariff τ and the carbon-targeted BCA τ_BA are equivalent here** (maps
  coincide to 3 decimals), because trade is ~100% dirty — both fall on the same
  shipments. The operative lever is the *total* border penalty on dirty imports, not
  its label. *(Do not report a τ-vs-τ_BA efficiency ranking; report equivalence.)*

### Exp 3 — minimum viable club (nucleation from a dirty start, h0=0.9)
A green club nucleates only above a critical founding coalition s0\*; below it the
coalition dissolves entirely (genuine bistability / tipping):

| Anchor φ | s0\* (μ/λ=1) | tipping s0 | transition band |
|---|---|---|---|
| 1500 (on boundary) | 0.86 | 0.92 | 0.12 |
| 1350 | 0.83 | 0.83 | 0.21 |
| 1200 (deeper green) | 0.69 | 0.71 | 0.38 |

The critical mass falls as fundamentals favour regulation (lower φ) and rises with
capital mobility; near the boundary the tipping is sharp, deeper in the green basin
it is more stochastic/path-dependent.

### Exp 4 — does the border charge lower the founding threshold?
**Yes.** s0\*(τ_BA) falls monotonically:

| τ_BA | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| s0\* | 0.97 | 0.93 | 0.90 | 0.83 | 0.66 |

- Slope **ds0\*/dτ_BA ≈ −0.27**; the charge brings the threshold from near-unanimity
  (0.97) to ~two-thirds (0.66) over [0,1] (not quite a bare majority within range).
- **Validation:** at τ_BA=0.5, s0\*=0.90, matching Exp 3's φ=1500 value (~0.86). ✓
- Raising τ_BA at the founding margin also **raises** the formed club's welfare
  (W_tot 957M → 971M → 973M). The enforcement lever and the founding lever coincide.

---

## 4. Welfare decomposition (normative layer)

Total social welfare W_tot = Σ_i W_i (what the analyst optimises) vs per-capita
W_i/P_i (what agents imitate). In the homogeneous-population baseline these are
proportional per jurisdiction, so any selected-vs-optimal gap is a **coordination
failure**, not a distributional artefact.

### Green club is globally welfare-superior — by +161%
On the speed map (fundamentals fixed; only μ/λ varies):

| Regime | W_tot | W per-capita |
|---|---|---|
| Green club | **931.7 M** | 16,992 |
| Transitional | 559.2 M | 10,199 |
| Race to bottom | **356.6 M** | 6,504 |

### Where the gap comes from (green vs race, totals)
| Component | Green | Race | Δ |
|---|---|---|---|
| Consumer surplus | 4.5 M | 5.3 M | −0.8 M |
| Fiscal revenue | 0.02 M | 0.06 M | −0.04 M |
| Host benefit (jobs/tax) | 1,052.9 M | 1,143.4 M | −90.5 M |
| Environmental damage | −125.7 M | −792.2 M | **+666.5 M** |
| **W_tot** | **931.7 M** | **356.6 M** | **+575 M** |

The green club forgoes ~91 M of dirty-production value but **avoids ~666 M of
damage** — avoided damage dwarfs everything. (Welfare is dominated by the
host-benefit and damage terms, both ∝ output × large coefficients; CS and fiscal
revenue are second-order.)

### The coordination-failure wedge (central normative result)
On the speed map every cell has identical fundamentals, so the welfare-maximising
regime is green club *everywhere*. Yet the dynamics **select race-to-bottom in 384 of
900 cells (43%)**, each forgoing **≈575 M of W_tot** — purely because capital moves
faster than institutions adapt. Myopic local imitation gets trapped in the
welfare-inferior basin even though the green club is globally optimal.

### Distribution: Pareto-dominant but unequal (Figure 4)
Per-capita welfare by bloc:

| Bloc | Green world | Race world |
|---|---|---|
| Strict bloc | 18,363 | 8,653 |
| Lax bloc | 9,530 | 5,114 |

The green-club outcome **Pareto-dominates** the race: every bloc is better off. But
the first-mover dynamics (Figure 4, contested baseline) show the surplus is shared
unequally:
- **Founders pay up front.** The founding strict bloc begins at the lowest welfare in
  the system (≈4,700) — hosting dirty firms that are taxed and fleeing.
- **Success:** founders climb to ≈18,200 as the club consolidates (s→1), but **late
  joiners end higher (≈35,800)** — a **first-mover penalty of ≈+17,500**: those who
  waited accumulated the fled industry and cleaned up last.
- **Failure (below threshold):** founders never recover (≈3,400, race-level) — going
  first is punished with total loss.

The laggards' rent is precisely the free-rider advantage of a lax jurisdiction hosting
the fled dirty industry; **the border carbon adjustment taxes it away**, which is why
(Exp 4) raising τ_BA lowers the founding threshold and raises welfare at the margin.

---

## 5. Headline takeaways (for the thesis)

1. **The race to the bottom is not inevitable** — within a contested band of
   fundamentals it is decided by the relative speed of capital vs institutions.
2. **The green club is welfare-superior by +161%** (W_tot 932 M vs 357 M), driven by
   ~666 M of avoided damage against only ~91 M of forgone dirty-production value.
3. **Coordination failure:** in 43% of the speed space the dynamics select the
   welfare-inferior regime despite identical fundamentals — ~575 M lost per cell.
4. **The transition is Pareto-improving but unequal:** every bloc gains versus the
   race, yet founders gain least and free-riding laggards gain most — a first-mover
   penalty (~+17,500 per-capita) that traps the system below the social optimum.
5. **A border carbon adjustment is the unifying lever:** a modest charge
   (~15–20% of the carbon price) sustains the club against fast capital (Exp 2),
   lowers the founding threshold from ~0.97 to ~0.66 (Exp 4), and raises welfare at
   the founding margin — because trade is ~100% dirty, a generic tariff and a
   carbon-targeted BCA are interchangeable; what matters is the total penalty on
   emissions-intensive imports.
