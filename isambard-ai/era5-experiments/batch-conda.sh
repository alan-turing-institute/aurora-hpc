#!/bin/bash
#SBATCH --job-name=aurora-conda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./batch-conda.sh

echo
echo "## Aurora conda script starting"

# Quit on error
set -e

echo
echo "## Initialising virtual environment"

source $HOME/miniforge3/bin/activate

conda create -n py311_torch26_cu126 -c conda-forge python=3.11 -y

conda activate py311_torch26_cu126

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install pynvml

conda create --name aurora_torch26 --clone py311_torch26_cu126

conda activate aurora_torch26

pushd ../../aurora/
python -m pip install .
popd

python -m pip install cdsapi matplotlib

echo
echo "## Verifying installation"
python verify.py

echo
echo "## Aurora runmodel script completed"

python -m pip freeze
