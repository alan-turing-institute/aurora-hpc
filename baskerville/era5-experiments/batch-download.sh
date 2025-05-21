#!/bin/bash
#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 0:10:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name auroria-prepare
#SBATCH --output log-prepare.txt

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

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load matplotlib/3.7.2-gfbf-2023a
module -q load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo
echo "## Initialising virtual environment"

python -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet cdsapi
pip install --quiet -e ../../aurora

echo
echo "## Downloading data"

python download.py

echo
echo "## Tidying up"

deactivate

echo
echo "## Aurora prepare script completed"
