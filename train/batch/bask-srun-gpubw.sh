#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# srun --qos turing --account usjs9456-ati-test --time 1:00:00 --nodes 1 --gpus 1 --cpus-per-gpu 36 --mem 16384 --pty /bin/bash
# ./bask-srun-gpubw.sh

echo "## Aurora GPU bandwidth script starting"

# Quit on error
# set -e

pushd ../scripts

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module -q load torchvision/0.15.2-foss-2022a-CUDA-11.7.0

echo "## Configuring environment"

export OMP_NUM_THREADS=1

echo "## Initialising virtual environment"

python3 -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet ../../.[bask]

echo "## Running model"

# Perform the prediction
python timing_gpu_bandwidth.py --device "gpu"

echo "## Tidying up"

deactivate
popd

echo "## Aurora GPU bandwidth script completed"
