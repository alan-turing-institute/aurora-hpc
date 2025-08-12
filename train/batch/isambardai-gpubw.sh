#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --job-name=gpubw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0


# Execute using:
# sbatch ./bask-local-gpubw.sh

echo "## Aurora GPU bandwidth script starting"

# Quit on error
set -e

echo "## Loading modules"

module purge

module load PrgEnv-gnu/8.5.0
module load craype-network-ofi
module load brics/nccl/v2.25.1-1-v1.6.x-r2
module load craype-accel-nvidia90

echo "## Configuring environment"

export OMP_NUM_THREADS=1

echo "## Initialising virtual environment"

source /home/u5q/tomas.u5q/miniforge3/bin/activate

conda activate aurora

# pip install --quiet typing-extensions==4.14.0

echo "## Running model"

# Perform the prediction
python ../scripts/timing_gpu_bandwidth.py --device "gpu"

echo "## Aurora GPU bandwidth script completed"
