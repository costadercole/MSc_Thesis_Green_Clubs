#!/bin/bash
#SBATCH --job-name=green_clubs_min_club
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/min_club_%j.out
#SBATCH --error=logs/min_club_%j.err

# 1. Load Python module
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# 2. Navigate to project directory
cd $SLURM_SUBMIT_DIR

# 3. Activate the pre-built virtual environment.
#    Build it ONCE interactively before submitting (see install steps below):
#      module load 2023; module load Python/3.11.3-GCCcore-12.3.0
#      python -m venv .venv && source .venv/bin/activate
#      python -m pip install -r requirements.txt
source .venv/bin/activate
python -c "import numpy, scipy, matplotlib, networkx" \
    || { echo "ERROR: .venv missing deps — build it first (see header)"; exit 1; }

# 4. Create logs directory if it doesn't exist
mkdir -p logs

# 5. Run Experiment B (minimum viable club) at three structural anchors, 6 seeds.
#    Produces only output/experiment_min_club/expB_s0_map[_phiXXXX].{png,csv} and
#    output/experiment_min_club/expB_hysteresis[_phiXXXX].{png,csv}; headlines print to this .out log.
for PHI in 1500 1350 1200; do
    echo "================ phi = $PHI ================"
    python experiments/experiment_min_club.py --phi $PHI --seeds 6
done
