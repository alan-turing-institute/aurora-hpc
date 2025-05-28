#!/bin/bash -l
#SBATCH --job-name=fine-tuning
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH -n 1   # Number of tasks (usually number of MPI ranks)
#SBATCH -c 96  # Number of cores per task
#SBATCH --gres=gpu:4 # Number of requested GPUs per node

set -o xtrace
set -o errexit

module purge
module load default-dawn

source ../environments/venv_3_11_9/bin/activate

export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

cd ../scripts/

ipython fine_tune.py
