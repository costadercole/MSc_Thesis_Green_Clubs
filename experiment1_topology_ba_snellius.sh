#!/bin/bash
#SBATCH --job-name=green_clubs_exp1_ba
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp1_ba_%j.out
#SBATCH --error=logs/exp1_ba_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
python -c "import numpy, scipy, matplotlib, networkx" \
    || { echo "ERROR: .venv missing deps — build it first"; exit 1; }
mkdir -p logs

# Robustness check: does the ER speed-map's mu~=lambda boundary hold on a
# Barabasi-Albert scale-free network? Runs BOTH m=1 (mean degree ~2) and
# m=2 (mean degree ~4) with m passed EXPLICITLY, bracketing the ER
# baseline's realised mean degree (~3.3) from below and above to match the
# ring_k2/ring_k4 degree levels. (NOTE: the previous version derived m from
# k via m = max(1, k // 2), which is not injective at low k — k=1 and k=2
# both mapped to m=1, so the old "ba1"/"ba2" runs silently built the SAME
# m=1 graph twice and produced identical output. Fixed by passing m
# explicitly; the corrected experiment script also asserts realised mean
# degree is within +/-0.3 of the intended target before running the sweep.)
# Same 30x30 mu x lambda grid, same T=2000, same contested point
# (phi=1500, delta_loc=1000, delta_glob=250), 8 seeds/cell (reported ER
# figure used 4). Produces output/experiment_topology/phase_speed_ba_m1.png/.csv,
# output/experiment_topology/phase_speed_ba_m2.png/.csv, and appends to
# output/experiment_topology/topology_robustness_summary.csv. Does not touch the ER baseline
# outputs (output/experiment_1/phase_speed.csv, output/experiment_1/phase_speed_final.png).
python experiments/experiment1_topology_robustness.py ba
