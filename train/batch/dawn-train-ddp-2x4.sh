#!/bin/bash -l
#SBATCH --job-name=2x4
#SBATCH --output=two_nodes_four_gpus.out
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH -c 24  # Number of cores per task
#SBATCH -N 2 # Number as nodes
#SBATCH --time 01:00:00
#SBATCH --exclusive

# 2 node, 4 GPUs
# For this we use two GPUs per node

#set -o xtrace
set -o errexit

module purge
module load default-dawn
module load lua
module load intel-oneapi-ccl/2021.14.0
module load intel-oneapi-mpi/2021.14.1
module load intel-oneapi-mkl/2025.0.1

# load intel oneapi compilers (gives us sycl-ls command)
module load intel-oneapi-compilers/2025.0.3/gcc/sb5vj5us

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

sycl-ls

mpirun -prepend-rand -n 4 bash -c 'stdbuf -o0 xpumcli dump -t 0,1 -m 0,1,2,5 -i 1 > gpu-${PMI_RANK}.out'
mpirun -prepend-rank -n 4 python train.py --xpu -d ../../dawn/era5/era_v_inf/

deactivate
popd
