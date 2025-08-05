#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# srun --qos turing --account usjs9456-ati-test --time 1:00:00 --nodes 1 --gpus 1 --cpus-per-gpu 36 --mem 16384 --pty /bin/bash
# ./bask-srun-calcs.sh

echo "## Aurora calculation testing script starting"

# Quit on error
set -e

pushd ../

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load matplotlib/3.7.2-gfbf-2023a
module -q load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "## Configuring environment"

export OMP_NUM_THREADS=1

echo "## Initialising virtual environment"

python3 -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet typing_extensions==4.14.1

echo "## Running calculations"

# Perform the calculations
python calculate.py -a "cuda" -p "CUDA " -o "calcs-bask-gpu.csv"
python calculate.py -a "cpu" -p "CPU (Baskerville) " -o "calcs-bask-cpu.csv"

echo "## Tidying up"

deactivate
popd

echo "## Aurora calculation testing script completed"
