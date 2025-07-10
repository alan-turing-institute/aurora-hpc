#!/bin/bash -l
#SBATCH --job-name=inference
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH --gres=gpu:1 # Number of requested GPUs per node
#SBATCH -N 1 # 1 node
#SBATCH --time 4:00:00 # HH:MM:SS

set -o xtrace
set -o errexit

module purge
module load default-dawn

source ../environments/venv_3_11_9/bin/activate

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

cd ../scripts/

# 6h timestep -> 4 steps/day -> 365*4 = 1460 steps/year
# removing 1 day (4 steps) so to allow for day 1 and day 365.
python inference.py -d ../era_5/era_v_inf/ -n 1456 --save 
