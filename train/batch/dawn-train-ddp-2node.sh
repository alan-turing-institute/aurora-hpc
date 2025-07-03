#!/bin/bash -l
#SBATCH --job-name=example-ddp
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH -n 8   # Number of tasks (usually number of MPI ranks)
#SBATCH -c 24  # Number of cores per task
#SBATCH --gres=gpu:4 # Number of requested GPUs per node
#SBATCH -N 2 # 2 nodes
#SBATCH --time 01:00:00

#set -o xtrace
set -o errexit

module purge
module load default-dawn
module load lua
module load intel-oneapi-ccl/2021.14.0
module load intel-oneapi-mpi/2021.14.1
module load intel-oneapi-mkl/2025.0.1

pushd ../scripts

source ../../dawn/environments/venv_3_11_9/bin/activate

# Merge tiles into full devices, for extra memory.
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# Avoid too many open file handles error.
ulimit -n 1000000

# Avoid mpi failing to init.
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=verbs

# Avoids segfaults, for some reason.
export ZES_ENABLE_SYSMAN=1

# Otherwise we're told to.
export CCL_ZE_IPC_EXCHANGE=sockets

mpirun -prepend-rank -n 8 python train.py --xpu -d ../../dawn/era5/era_v_inf/

deactivate
popd
