#!/bin/bash
#SBATCH --job-name="Aurora demo fine-tune"
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

# One-GPU fine-tuning run. We do one epoch of full fine-tuning
# with chunked activation checkpointing writing to a demo
# checkpoint directory and a demo TensorBoard log directory.

# Move to the root of the repo.
#cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"/..

export LD_LIBRARY_PATH="$PWD/gpu-env/lib"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

mkdir -p checkpoints/demo slurm_logs/nvidia

# GPU utilization logging.
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 \
  > "slurm_logs/nvidia/gpu-${SLURM_JOB_ID}.txt" &
DMON_PID=$!
trap 'kill "$DMON_PID" 2>/dev/null' EXIT

# NOTE: --nproc_per_node above 1 does NOT do real multi-GPU training today -
# train.py has no init_process_group/DistributedDataParallel/DistributedSampler,
# so extra ranks would redundantly retrain the full dataset rather than split
# it. TODO: implement real DDP before using this for multi-GPU wall-clock
# scaling. NPROC_PER_NODE defaults to 1, the only currently-correct value.
./gpu-env/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  train.py \
  --data_path ./data \
  --epochs 2 \
  --gpu \
  --checkpoint_dir checkpoints/demo \
  --log_dir "runs/demo-${SLURM_JOB_ID}" \
  "$@"
