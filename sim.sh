#!/bin/bash -l

#SBATCH --job-name=pycce_sim
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32 
#SBATCH --time=12:00:00             # shorter time = easier to schedule
#SBATCH --output=sim_%j.out
#SBATCH --error=sim_%j.err

cd "$SLURM_SUBMIT_DIR"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate pycce-env

echo "Running on host: $(hostname)"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "Working dir: $(pwd)"

srun python -u sim_temp_grid.py
