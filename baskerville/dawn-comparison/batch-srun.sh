#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# source ./batch-srun.sh

echo "## Aurora configuration script starting"

# Quit on error
set -e

if [ ! -d ../era5-experiments/downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load matplotlib/3.7.2-gfbf-2023a
module -q load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "## Initialising virtual environment"

python -m venv venv
. ./venv/bin/activate

#pip install --quiet --upgrade pip
#pip install --quiet cdsapi
#pip install --quiet -e ../../aurora

echo "## Aurora configuration script completed"
