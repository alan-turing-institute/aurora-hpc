#!/bin/bash -l
#SBATCH --job-name=2x8
#SBATCH --output results/two_nodes_eight_gpus_%A.out
#SBATCH --account=airr-p8-rcpp-dawn-gpu
#SBATCH --partition=pvc9 # Dawn PVC partition
#SBATCH --nodes 2 # Number as nodes
#SBATCH --gpus-per-node 4 # Number of requested GPUs per node
#SBATCH --ntasks-per-node 4 # MPI ranks per node
#SBATCH --time 01:00:00

# Execute using:
# sbatch ./dawn-train-ddp-2x8.sh

# 2 node, 8 GPUs
# For this we don't need to 'skip' any GPUs

pushd ../scripts

echo
echo "## Aurora DDP 1x1 training script starting"

#set -o xtrace
set -o errexit

pushd ../scripts

if [ ! -d ../../dawn/era5/era_v_inf ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo
echo "## Loading modules"

module purge
module load rhel9/default-dawn
module load intel-oneapi-mkl/2025.1.0
module load intel-oneapi-ccl/2021.15.0

echo
echo "## Configuring environment"

VENV_DIR=../../dawn/environments/venv_3_11_11_rhel9

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

echo
echo "## Initialising virtual environment"

source ${VENV_DIR}/bin/activate

echo
echo "## Details"
echo
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS per task: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: $(realpath ${PWD})"
echo "Location of venv: $(realpath ${VENV_DIR})"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "GPUs: ${SLURM_JOB_GPUS}"

echo
echo "## Running model"

# mpirun -host ${SLURM_JOB_NODELIST} bash -c 'stdbuf -o0 xpu-smi dump --rawdata --device $SLURM_JOB_GPUS -m 0,1,2,21,22 > gpu-${SLURM_JOB_ID}-${OMPI_COMM_WORLD_RANK}.txt' &

START=$(date +%s)
for i in {0..3}; do
  mpirun -prepend-rank -n 8 -ppn 4 python train.py --xpu -d ../../dawn/era5/era_v_inf/
done
END=$(date +%s)
ELAPSED=$((${END}-${START}))

echo
echo "## Details post"
echo
echo "Time completed: $(date --iso-8601=ns)"
echo "Epoch start: ${START}"
echo "Epoch end: ${END}"
echo "Elapsed: ${ELAPSED} seconds"

echo
echo "## Tidying up"

deactivate
popd

echo
echo "## Aurora DDP 1x1 training script completed"
