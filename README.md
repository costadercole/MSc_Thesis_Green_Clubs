# Green Clubs — MSc Thesis

An evolutionary agent-based model of **green club formation**: when do strict
environmental regulation clubs emerge and survive among trading jurisdictions,
and when does firm mobility trigger a regulatory race to the bottom?

## Model overview

`N` jurisdictions sit on a network (ring lattice, Erdős–Rényi, or
Barabási–Albert). Each jurisdiction is either **strict** (carbon tax `t`,
tariff `tau` and border carbon adjustment `tau_BA` on imports from lax
neighbours) or **lax**. `M` firms live in the jurisdictions and are either
**high-emission (H)** or **low-emission (L)** (marginal costs `c_H < c_L`).

Each period:

1. **Goods market** (`model/market.py`) — Cournot competition in every
   jurisdiction's market, with trade to network neighbours. Delivered cost adds
   transport cost `g`, plus `tau` (and `tau_BA` for H-firms) on lax→strict
   exports. Solved by monotone elimination of unprofitable firms.
2. **Welfare** (`model/jurisdictions.py`) — per-capita welfare = consumer
   surplus + fiscal revenue (tax + tariff + BCA) + host benefit
   `phi`·(local output) − local damage `delta_loc`·(local H output)
   − global damage `delta_glob`·(total H output). The `phi` vs `delta_loc`
   trade-off is what makes hosting dirty industry attractive or not.
3. **Firm relocation** (`model/firms.py`, rate `mu`) — H-firms move towards the
   most profitable neighbouring jurisdiction (pollution-haven channel).
4. **Firm type switching** (rate `nu`) — firms imitate the emission type of a
   randomly sampled firm via a Fermi rule on profit differences.
5. **Policy imitation** (rate `lam`) — jurisdictions imitate a random
   neighbour's policy via a Fermi rule on per-capita welfare differences.

Steady-state outcomes are classified by the high-emission firm share `h` and the
strict jurisdiction share `s` into **green club** (low `h`, high `s`),
**race to bottom** (high `h`, low `s`), or **transitional/contested**
(`analysis/metrics.py`). The key order parameter is `R = s − h`.

The central result explored by the experiments: the outcome depends on the
*relative speed* of firm mobility vs institutional adaptation (`mu/lam`), on
the economic fundamentals (`phi`, `delta_loc`), and on the policy levers
(`tau`, `tau_BA`).

## Repository layout

```
params.py                  # all parameters + calibrated BASELINE dataclass
run_and_plot.py            # single baseline run → diagnostic plots in output/
model/
  network.py               # ring / ER / BA network + row-normalised weights W
  market.py                # Cournot equilibrium, delivered costs, firm profits
  firms.py                 # firm init, relocation (mu), type imitation (nu)
  jurisdictions.py         # fiscal revenue, per-capita welfare, policy imitation (lam)
  simulation.py            # main loop (burn-in + steps 1–5 above)
analysis/
  metrics.py               # steady state, outcome classification
  welfare.py               # welfare decomposition (CS / fiscal / host / damage)
  plots.py                 # time series, phase portrait, firm distribution
calibration/
  calibration.py           # phase-1 moments: leakage, import penetration, price CV
  run_calibration.py       # grid search over (delta_c, t_ratio, g)
  calibrate_damage.py      # phase-2: strict win-rate over (phi, delta_loc)
experiments/
  sweep_lib.py             # shared parallel grid-sweep machinery + welfare logging
  phase_diagram.py         # Experiment 1: speed (mu×lam) & structural (phi×delta_loc) maps
  replot_phase.py          # thesis-ready re-plots of Experiment 1 from saved CSVs
  experiment1_topology_robustness.py  # Exp 1 speed map on ring/BA topologies
  experiment_tauBA.py      # Experiment A: CBAM lever maps (mu×tau_BA, mu×tau)
  experiment_min_club.py   # Experiment B: nucleation / minimum viable club (mu×s0)
  experiment4_bca_nucleation.py       # Experiment 4: does tau_BA lower the founding threshold?
  experiment1_welfare.py   # Exp 1 rerun with welfare logging (feeds welfare figures)
  plot_welfare_speed_maps.py          # welfare map + coordination-failure wedge
  figure4_first_mover.py   # founding bloc vs late joiners welfare trajectories
tests/                     # pytest suite (Cournot equilibrium, type update, welfare accounting)
*_snellius.sh              # SLURM job scripts for the Snellius cluster
```

## Output layout

Each experiment writes its CSVs and figures to its own subfolder of `output/`:

| Folder | Produced by | Contents |
|---|---|---|
| `output/experiment_1/` | `phase_diagram.py`, `replot_phase.py` | speed & structural phase maps (`phase_speed*`, `phase_structural*`) |
| `output/experiment_topology/` | `experiment1_topology_robustness.py` | speed maps on ER / ring / BA (`phase_speed_<topo>*`, summary CSV) |
| `output/experiment_tauBA/` | `experiment_tauBA.py` | CBAM lever maps (`expA_tauBA_map*`, `expA_tau_map*`) |
| `output/experiment_min_club/` | `experiment_min_club.py` | nucleation maps & hysteresis (`expB_s0_map*`, `expB_hysteresis*`) |
| `output/experiment_4/` | `experiment4_bca_nucleation.py` | founding threshold vs border charge (`exp4_*`) |
| `output/experiment_welfare/` | `experiment1_welfare.py`, `plot_welfare_speed_maps.py`, `figure4_first_mover.py` | welfare-logged Exp-1 maps, welfare/wedge maps, first-mover figure |
| `output/calibration/` | `run_calibration.py`, `calibrate_damage.py` | calibration grids, baseline params, damage calibration |

Diagnostic plots from `run_and_plot.py` go to `output/` directly.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Single baseline simulation + diagnostic plots
python run_and_plot.py                      # or e.g. --mu 3.0 --lam 0.5 --T 2000

# Tests
python -m pytest tests/ -q

# Experiment 1 — phase diagrams (900 cells × seeds; use a cluster or be patient)
python experiments/phase_diagram.py structural
python experiments/phase_diagram.py speed
python experiments/replot_phase.py structural   # polished figures from saved CSVs
python experiments/replot_phase.py speed

# Topology robustness (er | ring | ba | all)
python experiments/experiment1_topology_robustness.py all

# Policy experiments (all support --quick for a coarse fast pass)
python experiments/experiment_tauBA.py --quick
python experiments/experiment_min_club.py --quick
python experiments/experiment4_bca_nucleation.py --quick

# Welfare layer
python experiments/experiment1_welfare.py both
python experiments/plot_welfare_speed_maps.py
python experiments/figure4_first_mover.py

# Calibration
python calibration/run_calibration.py
python calibration/calibrate_damage.py
```

On the Snellius cluster, submit the corresponding `*_snellius.sh` job script
with `sbatch` (each activates `.venv` and writes logs to `logs/`).
