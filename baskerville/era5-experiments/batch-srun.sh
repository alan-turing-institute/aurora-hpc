#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# . ./batch-srun.sh

echo "## Aurora srun script starting"

if [ ! -d downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load matplotlib/3.7.2-gfbf-2023a
module -q load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "## Configuring environment"

export OMP_NUM_THREADS=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9724

echo "## Initialising virtual environment"

python -m venv venv
. ./venv/bin/activate

#pip install --quiet --upgrade pip
#pip install --quiet cdsapi
#pip install --quiet -e ../../aurora

echo "## Aurora srun script completed"
