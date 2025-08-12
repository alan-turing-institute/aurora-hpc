#!/bin/bash
#SBATCH --job-name=aurora-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=0

# Execute using:
# sbatch ./isambardai-train-fsdp.sh

echo
echo "## Aurora fine-tuning script starting"

# Quit on error
set -e

if [ ! -d ../../downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo
echo "## Loading modules"

module purge

module load PrgEnv-gnu/8.5.0
module load craype-network-ofi
module load brics/nccl/v2.25.1-1-v1.6.x-r2
module load craype-accel-nvidia90


echo
echo "## Configuring environment"

export PRIMARY_PORT=$((16384 + $RANDOM % 16384))
export PRIMARY_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=1

echo
echo "## Initialising virtual environment"

source /home/u5q/tomas.u5q/miniforge3/bin/activate

conda activate aurora

echo
echo "## Details"
echo
echo "Nodes: ${SLURM_GPUS_PER_NODE}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Primary address: ${PRIMARY_ADDR}"
echo "Primary port: ${PRIMARY_PORT}"

echo
echo "## Running model"

# Track GPU and CPU metrics
nvidia-smi dmon -o TD -s puct -d 1 > logs/aurora-finetune-gpu.txt &
vmstat -t 1 > logs/aurora-finetune-cpu.txt &

# Perform the prediction
srun bash -c \
    'python -m torch.distributed.run \
    --nnodes ${SLURM_JOB_NUM_NODES} \
    --nproc-per-node ${SLURM_GPUS_PER_NODE} \
    --master_addr ${PRIMARY_ADDR} \
    --master_port ${PRIMARY_PORT} \
    --node_rank ${SLURM_NODEID} \
    ../scripts/train.py \
    -d ../../downloads'

echo
echo "## Tidying up"

echo
echo "## Aurora fine-tuning script completed"
