#!/bin/bash
#SBATCH --job-name=aurora-download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./batch-download.sh

echo
echo "## Aurora download script starting"

# Quit on error
set -e

export CDSAPI_RC=$PWD/cdsapi.config

if [ ! -f $CDSAPI_RC ]; then
  echo "Please create a CDSAPI configuration file at $CDSAPI_RC. See https://github.com/ecmwf/cdsapi?tab=readme-ov-file#configure"
  exit 1
fi

echo
echo "## Initialising virtual environment"

source $HOME/miniforge3/bin/activate

conda activate aurora_torch26

echo
echo "## Downloading data"

python download.py

echo
echo "## Aurora download script completed"
