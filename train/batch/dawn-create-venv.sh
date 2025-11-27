#!/bin/bash -l
#SBATCH --job-name venv
#SBATCH --output results/create-venv-%A.out
#SBATCH --account airr-p8-rcpp-dawn-gpu
#SBATCH --partition pvc9 # Dawn PVC partition
#SBATCH --cpus-per-task 24  # Number of cores per task
#SBATCH --nodes 1 # Number as nodes
#SBATCH --gpus-per-node 1 # Number of requested GPUs per node
#SBATCH --ntasks-per-node 1 # MPI ranks per node
#SBATCH --time 01:00:00

# Execute using:
# sbatch ./dawn-create-venv.sh

echo
echo "## Aurora create virtual environment script starting"

# Quit on error
set -e

pushd ../scripts

echo
echo "## Loading modules"

module purge
module load rhel9/default-dawn
module load intel-oneapi-mkl/2025.1.0
module load intel-oneapi-ccl/2021.15.0

echo
echo "## Configuring environment"

VENV_DIR=../../dawn/environments/venv_3_11_11_rhel9

echo
echo "## Initialising virtual environment"

python3.11 -m venv $VENV_DIR
. ${VENV_DIR}/bin/activate

echo
echo "## Details"
echo
echo "Date: $(date)"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "CPUS per task: ${SLURM_CPUS_PER_TASK}"
echo "Working directory: $(realpath ${PWD})"
echo "Location of venv: $(realpath $VENV_DIR)"

echo
echo "## Installing packages"

START=$(date +%s)
pip install --upgrade pip
pip install -e ../../.[dawn]
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
pip install --trusted-host pytorch-extension.intel.com intel-extension-for-pytorch==2.7.10+xpu oneccl_bind_pt==2.7.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
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
echo "## Aurora create virtual environmnent script completed"
