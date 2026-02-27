#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --time 10:00:0
#SBATCH --gpus 1

echo
echo "Aurora fine-tuning script starting"

# Quit on error
set -e

module load cuda/12.6

# Activate the environment
. ../../isambard/environments/venv/bin/activate

# Show the Python location for the logs.
echo "$(which python)"

pushd ../scripts

# Nvidia-SMI GPU stats.
nvidia-smi dmon -o TD -s puct -d 1 > log-train-gpu.txt &

# Todo (on Isambard?)
#vmstat -t 1 -y > log-train-cpu.txt &

WORLD_SIZE=1
export WORLD_SIZE
RANK=0
export RANK
LOCAL_RANK=0
export LOCAL_RANK
MASTER_ADDR=localhost
export MASTER_ADDR
MASTER_PORT=8334
export MASTER_PORT

RUN_TS="$(date +%Y%m%dT%H%M%S)"
JOB_TAG="${SLURM_JOB_ID:-local}"
TB_BASE_LOGDIR="${TB_BASE_LOGDIR:-runs/aurora_train}"
TB_RUN_LOGDIR="${TB_BASE_LOGDIR}/${RUN_TS}_job${JOB_TAG}"
echo "TensorBoard logdir: ${TB_RUN_LOGDIR}"

python train.py \
  --download_path "../../../datasets/era5/2023-01" \
  --epochs 2 \
  --model_size "small" \
  --learning_rate 0.0001 \
  --target_global_batch 1 \
  --tb_logdir "${TB_RUN_LOGDIR}"

echo "Aurora fine-tuning finished"
