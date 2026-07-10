#!/bin/bash
# Call our Python training script.

set -e

# All options other than --nproc-per-node are passed on to train.py
nproc_per_node="1"
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc-per-node)
      nproc_per_node="$2"
      shift 2
      ;;
    --nproc-per-node=*)
      nproc_per_node="${1#*=}"
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# Activate our environment.
eval "$(mamba shell hook --shell bash)"
mamba activate ./gpu-env

# So we can tail logs better.
PYTHONUNBUFFERED=1
export PYTHONBUFFERED

# As per the recommendations in the Aurora docs.
PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
export PYTORCH_CUDA_ALLOC_CONF

set -e

# Train.
# Note that this method of calling torchrun is required if the torchrun executable
# shebang doesn't match how we've mounted the working directory in our container.
python -m torch.distributed.run --standalone --nproc_per_node=$nproc_per_node train.py \
  --data_path=./data \
  --log_level=DEBUG \
  --chunk \
  "${args[@]}"
