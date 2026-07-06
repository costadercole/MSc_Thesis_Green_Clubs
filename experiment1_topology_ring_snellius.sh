#!/bin/bash
#SBATCH --job-name=green_clubs_exp1_ring
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp1_ring_%j.out
#SBATCH --error=logs/exp1_ring_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
python -c "import numpy, scipy, matplotlib, networkx" \
    || { echo "ERROR: .venv missing deps — build it first"; exit 1; }
mkdir -p logs

# Robustness check: does the ER speed-map's mu~=lambda boundary hold on
# k-regular ring lattices? Runs BOTH ring_k2 (k=2, exact degree 2) and
# ring_k4 (k=4, exact degree 4), bracketing the ER baseline's realised mean
# degree (~3.3) from below and above so the ring-vs-BA comparison at each
# degree level is degree-matched. (NOTE: k=3 was dropped — Watts-Strogatz
# with p=0 connects k//2 neighbours per side, so odd k does not give an
# exact degree; k=3 silently produced degree 2, not 3. Only even k is used
# now, and the corrected experiment script asserts realised mean degree is
# within +/-0.3 of the intended target before running the sweep.)
# Same 30x30 mu x lambda grid, same T=2000, same contested point
# (phi=1500, delta_loc=1000, delta_glob=250), 8 seeds/cell (reported ER
# figure used 4). Produces output/experiment_topology/phase_speed_ring_k2.png/.csv,
# output/experiment_topology/phase_speed_ring_k4.png/.csv, and appends to
# output/experiment_topology/topology_robustness_summary.csv. Does not touch the ER baseline
# outputs (output/experiment_1/phase_speed.csv, output/experiment_1/phase_speed_final.png).
python experiments/experiment1_topology_robustness.py ring
