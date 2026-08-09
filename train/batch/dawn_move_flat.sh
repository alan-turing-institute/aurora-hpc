#!/bin/bash -l
#SBATCH --job-name=1x1
#SBATCH --output=dawn_move.out
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH -c 24  # Number of cores per task
#SBATCH -N 1 # Number as nodes
#SBATCH --gres=gpu:1 # Number of requested GPUs per node
#SBATCH --ntasks-per-node=1 # MPI ranks per node
#SBATCH --time 00:10:00

# 1 node, 1 GPU
# For this we don't need to 'skip' any GPUs

#set -o xtrace
set -o errexit

#module purge
#module load default-dawn
#module load lua
#module load intel-oneapi-ccl/2021.14.0
#module load intel-oneapi-mpi/2021.14.1
#module load intel-oneapi-mkl/2025.0.1

pushd ../scripts

#source ../../dawn/environments/venv_3_11_9/bin/activate

# Merge tiles into full devices, for extra memory.
echo "Setting FLAT"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
#echo "Setting COMPOSITE"
#export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# Avoid too many open file handles error.
ulimit -n 1000000

# Avoid mpi failing to init.
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=verbs

# Avoids segfaults, for some reason.
export ZES_ENABLE_SYSMAN=1

# Otherwise we're told to.
export CCL_ZE_IPC_EXCHANGE=sockets

echo "Starting move"
#python -m cProfile -o train_to.pstats train.py --xpu --xpu-optimize -d ../../dawn/era5/era_v_inf/
python run_aurora_small.py

#deactivate
popd
