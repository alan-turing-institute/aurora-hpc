#!/bin/bash
#SBATCH --job-name=aurora-runmodel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./batch-runmodel.sh

echo
echo "## Aurora runmodel script starting"

# Quit on error
set -e

if [ ! -d downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo 
echo "## Loading modules"

module purge

module load PrgEnv-gnu/8.5.0
module load craype-network-ofi
module load brics/nccl/v2.25.1-1-v1.6.x-r2
module load craype-accel-nvidia90

echo
echo "## Initialising virtual environment"

source /home/u5q/tomas.u5q/miniforge3/bin/activate

conda activate aurora

echo
echo "## Running model"

# Track GPU and CPU metrics
nvidia-smi dmon -o TD -s puct -d 1 > logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu.txt &
vmstat -t 1 > logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-cpu.txt &

# Perform the prediction
python runmodel.py

echo
echo "## Aurora runmodel script completed"
