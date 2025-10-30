#!/bin/bash -l
# vim: et:ts=4:sts=4:sw=4
#SBATCH --qos turing
#SBATCH --account usjs9456-ati-test
#SBATCH --time 1:00:0
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --mem 0
#SBATCH --constraint=a100_80
#SBATCH --job-name aurora-train
#SBATCH --output bask-encoder-%a.txt

# Execute using:
# sbatch --array=5-29 ./bask-train-ddp-1x1.sh

# 1 node, 1 GPU
# For this we don't need to 'skip' any GPUs

#set -o xtrace
set -o errexit

pushd ../../scripts

if [ ! -d ../../downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo
echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module -q load torchvision/0.15.2-foss-2022a-CUDA-11.7.0

echo
echo "## Configuring environment"

export PRIMARY_PORT=$((16384 + $RANDOM % 16384))
export PRIMARY_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=1
export ENCODER_DEPTH=${SLURM_ARRAY_TASK_ID}

echo
echo "## Initialising virtual environment"

python -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet ../../.[bask]

echo
echo "## Details"
echo
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Primary address: ${PRIMARY_ADDR}"
echo "Primary port: ${PRIMARY_PORT}"
echo "Encoder depth: ${ENCODER_DEPTH}"

echo
echo "## Running model"

# Track GPU and CPU metrics
#nvidia-smi dmon -o TD -s puct -d 1 > log-train-gpu.txt &
#vmstat -t 1 -y > log-train-cpu.txt &

# Perform the prediction
# Repeat this 4 times so we get better logs
srun bash -c \
    'python -m torch.distributed.run \
    --nnodes ${SLURM_JOB_NUM_NODES} \
    --nproc-per-node ${SLURM_GPUS_PER_NODE} \
    --master_addr ${PRIMARY_ADDR} \
    --master_port ${PRIMARY_PORT} \
    --node_rank ${SLURM_NODEID} \
    train_ed.py \
    --download_path ../../downloads \
    --encoders ${ENCODER_DEPTH} \
    --grad_accum 8'

echo
echo "## Tidying up"

deactivate
popd
