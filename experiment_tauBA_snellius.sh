#!/bin/bash
#SBATCH --job-name=green_clubs_tauBA
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tauBA_%j.out
#SBATCH --error=logs/tauBA_%j.err

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

# 5. Run Experiment A (CBAM): mu x tau_BA and mu x tau, 6 seeds, lever range [0,1]
#    Produces only output/expA_tauBA_map.{png,csv} and output/expA_tau_map.{png,csv};
#    headline numbers print to this job's .out log.
python experiments/experiment_tauBA.py --seeds 6 --tauba_max 1.0 --tau_max 1.0
