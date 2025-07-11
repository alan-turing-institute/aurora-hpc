#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 0:10:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --mem 32768
#SBATCH --job-name auroria-comparison
#SBATCH --output log-comparison.txt

# Execute using:
# sbatch ./batch-comparison.sh

echo
echo "## Aurora comparison script starting"

# Quit on error
set -e

if [ ! -d ../era5-experiments/downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
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
pip install --quiet -e ../../aurora-hpc

echo
echo "## Running model"

# Track GPU and CPU metrics
nvidia-smi dmon -o TD -s puct -d 1 > log-comparison-gpu.txt &
vmstat -t 1 -y > log-comparison-cpu.txt &

# Perform the prediction
python inference-timing.py --nsteps 28 --save --output_file preds-bask.pkl

# Generate graphs
python compare-results.py

echo
echo "## Tidying up"

deactivate

echo
echo "## Aurora comparison script completed"
