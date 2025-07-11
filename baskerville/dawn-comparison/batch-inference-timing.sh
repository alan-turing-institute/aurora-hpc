#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 0:10:0
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-gpu 36
#SBATCH --constraint=a100_80
#SBATCH --job-name auroria-comparison
#SBATCH --output log-timing.txt

# Execute using:
# sbatch ./batch-inference-timing.sh

echo
echo "## Aurora inference timing script starting"

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
# do this 4 times, once per GPU
for i in {0..4}; do
    CUDA_VISIBLE_DEVICES=$i python inference-timing.py -n 28 --save -o preds_$i.pkl > inference_28_steps_$i.txt &
done

wait 

echo
echo "## Tidying up"

deactivate

echo
echo "## Aurora inference-timing script completed"
