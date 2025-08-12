#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --job-name=diskbw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./bask-local-diskbw.sh

echo "## Aurora disk bandwidth script starting"

# Quit on error
set -e

if [ ! -d ../../downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

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

# python -m pip install --quiet --upgrade pip
# python -m pip install "pillow<11.0"
# python -m pip install --quiet xarray
python -m pip install dask
# python -m pip install --quiet typing-extensions
# python -m pip install --quiet -e ../../aurora

# python -m pip freeze 

echo "## Running model"

# Perform the prediction
python ../scripts/timing_data_transfer.py -d ../../downloads --dask

echo "## Aurora disk bandwidth script completed"
