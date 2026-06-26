#!/bin/bash
#SBATCH --job-name=green_clubs_exp4
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp4_%j.out
#SBATCH --error=logs/exp4_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
python -c "import numpy, scipy, matplotlib, networkx" \
    || { echo "ERROR: .venv missing deps — build it first"; exit 1; }
mkdir -p logs
python experiments/experiment4_bca_nucleation.py
