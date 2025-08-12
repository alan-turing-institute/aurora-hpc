#!/bin/bash
#SBATCH --job-name=aurora-conda
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

conda create --name aurora --clone pytorch_env

conda activate aurora

python -m pip install --upgrade pip --no-cache-dir
python -m pip install -e ../../aurora --no-cache-dir
python -m pip install pynvml cdsapi matplotlib --no-cache-dir

echo
echo "## Verifying installation"
python verify.py

echo
echo "## Aurora runmodel script completed"

python -m pip freeze
