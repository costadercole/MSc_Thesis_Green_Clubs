#!/bin/bash
#SBATCH --job-name=green_clubs_calibration
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/calibration_%j.out
#SBATCH --error=logs/calibration_%j.err

# 1. Load Python module
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# 2. Navigate to project directory
cd $SLURM_SUBMIT_DIR

# 3. Install dependencies into a local virtual environment (first run only)
if [ ! -d ".venv" ]; then
    python -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# 4. Activate environment
source .venv/bin/activate

# 5. Create logs directory if it doesn't exist
mkdir -p logs

# 6. Run the calibration + bifurcation search
python calibration/calibrate_then_bifurcate.py
