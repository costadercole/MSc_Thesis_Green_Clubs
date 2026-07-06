#!/bin/bash
#SBATCH --job-name=green_clubs_exp1_topo_all
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp1_topo_all_%j.out
#SBATCH --error=logs/exp1_topo_all_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
python -c "import numpy, scipy, matplotlib, networkx" \
    || { echo "ERROR: .venv missing deps — build it first"; exit 1; }
mkdir -p logs

# Full topology-robustness comparison in one job: reruns the ER reference
# at the same 8-seeds/cell protocol as the structured topologies (the
# *reported* thesis figure output/experiment_1/phase_speed_final.png used 4 seeds/cell
# and is left untouched), plus ring_k2, ring_k4, ba_m1, ba_m2 -- the four
# structured topologies that bracket ER's realised mean degree (~3.3) from
# below (~2) and above (~4).
#
# Rebuilds output/experiment_topology/topology_robustness_summary.csv FROM SCRATCH (5 rows: er,
# ring_k2, ring_k4, ba_m1, ba_m2) and prints the verification block (degree
# guards, boundary-file distinctness, row count, median mu*/lambda side per
# topology). Does not touch output/experiment_1/phase_speed.csv or
# output/experiment_1/phase_speed_final.png.
python experiments/experiment1_topology_robustness.py all
