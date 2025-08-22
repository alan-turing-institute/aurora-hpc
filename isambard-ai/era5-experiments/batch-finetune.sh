#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --job-name=aurora-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./batch-finetune.sh

echo
echo "## Aurora fine-tuning script starting"

# Quit on error
set -e

if [ ! -d ../../downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo
echo "## Initialising virtual environment"

source $HOME/miniforge3/bin/activate

conda activate aurora_torch26

echo
echo "## Running model"

# Track GPU and CPU metrics
nvidia-smi dmon -o TD -s puct -d 1 > logs/aurora-finetune-gpu.txt &
vmstat -t 1 > logs/aurora-finetune-cpu.txt &

# Perform the prediction
python finetune.py

echo
echo "## Tidying up"

deactivate

echo
echo "## Aurora fine-tuning script completed"
