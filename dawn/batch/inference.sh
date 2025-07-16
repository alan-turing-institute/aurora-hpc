#!/bin/bash -l
#SBATCH --job-name=inference
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH --gres=gpu:4 # Number of requested GPUs per node
#SBATCH -N 1 # 1 node
#SBATCH --time 4:00:00 # HH:MM:SS

set -o xtrace
set -o errexit

module purge
module load default-dawn

source ../environments/venv_3_11_9/bin/activate

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

cd ../scripts/

# run on each GPU
for i in {0..3}; do
  ZE_AFFINITY_MASK=$i python inference.py --d ../era5/era_v_inf/ -n 28 --save -o preds_$i.pkl > inference_28_steps_$i.txt &
done

wait
