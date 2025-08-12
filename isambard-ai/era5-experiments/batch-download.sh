#!/bin/bash
#SBATCH --job-name=aurora-prepare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./batch-prepare.sh

echo
echo "## Aurora prepare script starting"

# Quit on error
set -e

export CDSAPI_RC=$PWD/cdsapi.config

if [ ! -f $CDSAPI_RC ]; then
  echo "Please create a CDSAPI configuration file at $CDSAPI_RC. See https://github.com/ecmwf/cdsapi?tab=readme-ov-file#configure"
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
echo "## Downloading data"

python download.py

echo
echo "## Aurora prepare script completed"
