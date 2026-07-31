#!/bin/bash
#SBATCH --job-name="Aurora benchmark"
#SBATCH --time=00:30:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

# Short benchmark run: a handful of optimizer steps, no checkpoint save, for
# measuring perf/* and gpu_memory/* TensorBoard scalars plus the
# benchmark_summary.json written to the run's log_dir. This is NOT a real
# training run - use gpu_train.sbatch for that.
#
# TF32 and BF16 autocast are both on by default in train.py now (see
# job 42964052 / runs/benchmark-42964052-*: ~4.7x faster and lower peak GPU
# memory than fp32, with no accuracy validation beyond sane, non-NaN losses
# over 5 steps - see [[aurora-hpc-tf32-autocast-defaults]] memory note before
# relying on this for real training quality). This script runs a single
# default-config benchmark; pass --no-tf32/--no-autocast (or add a second
# run_benchmark call below) for an ad hoc comparison.
#
# Note: AuroraDataset eagerly loads the whole month of data (~36GB) up front
# regardless of --max_steps, which took ~6.5 minutes in past runs - that cost
# is unavoidable here and dominates wall time, not the (now capped) training
# loop itself. --time is sized to give headroom above that.

export LD_LIBRARY_PATH="$PWD/gpu-env/lib"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

mkdir -p checkpoints/benchmark slurm_logs/nvidia

# GPU utilization logging.
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 \
  > "slurm_logs/nvidia/gpu-${SLURM_JOB_ID}.txt" &
DMON_PID=$!
trap 'kill "$DMON_PID" 2>/dev/null' EXIT

# Defaults to the standard-size model, since that's what's actually trained
# and what we care about the perf/memory numbers for. To benchmark the small
# model instead (e.g. a quick sanity check of the instrumentation itself),
# set MODEL_ARGS=(--use_small) below.
MODEL_ARGS=()

# --hf_offline below assumes the checkpoint(s) for whichever model size you
# pick have already been downloaded at least once (true as of this writing
# for the standard model). If you hit a cache-miss error, drop --hf_offline
# from run_benchmark for one run to (re-)populate the cache.

# NOTE: --nproc_per_node above 1 does NOT do real multi-GPU training today.
# train.py has no init_process_group/DistributedDataParallel/DistributedSampler,
# so every rank would redundantly train on the full dataset, most likely all
# mapped onto the same GPU, racing to write the same checkpoint/log paths.
# TODO: come back to this if/when we actually want multi-GPU wall-clock
# scaling - needs real DDP wiring, not just launching more ranks.
# NPROC_PER_NODE defaults to 1, the only currently-correct value.
run_benchmark() {
  local label="$1"
  shift
  ./gpu-env/bin/python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE:-1}" \
    train.py \
    --data_path ./data \
    "${MODEL_ARGS[@]}" \
    --epochs 1 \
    --max_steps "${MAX_STEPS:-5}" \
    --skip_checkpoint \
    --gpu \
    --hf_offline \
    --checkpoint_dir checkpoints/benchmark \
    --log_dir "runs/benchmark-${SLURM_JOB_ID}-${label}" \
    "$@"
}

run_benchmark default "$@"
