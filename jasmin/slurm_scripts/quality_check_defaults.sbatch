#!/bin/bash
#SBATCH --job-name="Aurora quality check (tf32+autocast defaults)"
#SBATCH --time=01:15:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

# Companion to quality_check_baseline.sbatch: fine-tuning run using train.py's
# current defaults (TF32 + BF16 autocast both on), same --seed and epoch
# count, to compare train/loss curves and eval RMSE (via evaluate.sbatch
# afterwards) against the full-precision baseline. See
# [[aurora-hpc-tf32-autocast-defaults]] memory note - this is the validation
# that was still outstanding there.
#
# Real checkpoints are saved here (no --skip_checkpoint, no --max_steps) -
# unlike benchmark_train.sbatch this is NOT a throwaway run. Deliberately
# passes no --no-tf32/--no-autocast: this run tracks whatever train.py's
# defaults are, so it stays a fair "what a user gets by default" comparison
# even if those defaults change later.

export LD_LIBRARY_PATH="$PWD/gpu-env/lib"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

mkdir -p checkpoints/quality-defaults slurm_logs/nvidia

# GPU utilization logging.
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 \
  > "slurm_logs/nvidia/gpu-${SLURM_JOB_ID}.txt" &
DMON_PID=$!
trap 'kill "$DMON_PID" 2>/dev/null' EXIT

# NOTE: --nproc_per_node above 1 does NOT do real multi-GPU training today -
# train.py has no init_process_group/DistributedDataParallel/DistributedSampler.
# NPROC_PER_NODE defaults to 1, the only currently-correct value.
./gpu-env/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  train.py \
  --data_path ./data \
  --epochs "${EPOCHS:-2}" \
  --seed "${SEED:-42}" \
  --hf_offline \
  --gpu \
  --checkpoint_dir checkpoints/quality-defaults \
  --log_dir "runs/quality-defaults-${SLURM_JOB_ID}" \
  "$@"
