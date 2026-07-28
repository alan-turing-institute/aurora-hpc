#!/bin/bash
#SBATCH --job-name="Aurora evaluation"
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err

# Evaluate base Aurora vs the fine-tuned checkpoint.
# Results land in eval_results/ for the demo notebook to plot.

# Extra arguments are passed straight to evaluate.py, e.g.
#   sbatch slurm_scripts/evaluate.sbatch --steps 8 --inits 0,32

export LD_LIBRARY_PATH="$PWD/mamba-env/lib"
export PYTHONUNBUFFERED=1

./mamba-env/bin/python evaluate.py \
  --checkpoint checkpoints/gpu/epoch-0004.pt \
  --models base,finetuned,persistence,climatology \
  --steps 20 \
  --inits 0,16,32,48,64,80 \
  --device cuda \
  --out eval_results \
  "$@"
