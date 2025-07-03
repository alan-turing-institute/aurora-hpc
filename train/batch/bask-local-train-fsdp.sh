#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# sbatch ./bask-local-train-fsdp.sh

echo "## Aurora fine-tuning script starting"

# Quit on error
set -e

pushd ../scripts

if [ ! -d ../../downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module -q load torchvision/0.15.2-foss-2022a-CUDA-11.7.0

echo "## Configuring environment"

export OMP_NUM_THREADS=1

echo "## Initialising virtual environment"

python -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet xarray==2023.1.0
pip install --quiet cdsapi
pip install --quiet -e ../../aurora

echo "## Running model"

# Perform the prediction
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=28743 python train.py -d ../../downloads

echo "## Tidying up"

deactivate
popd

echo "## Aurora fine-tuning script completed"
