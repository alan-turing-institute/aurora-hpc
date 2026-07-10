# Aurora fine-tuning on JASMIN

Fine-tuning of Microsoft's Aurora weather model on regridded CMIP6 output (MPI-ESM, 0.25°), plus evaluation and a demo notebook.

On JASMIN, interactive work can be done on the sci-vm-0x machines or via slurm interactive jobs.
Batch jobs can be submitted via the slurm_script/ sbatch files, including jobs on ORCHID (JASMIN's A100 cluster).
Start with the pre-rendered `demo.ipynb` for a tour.

## Layout

- `train.py` - fine-tuning entry point.
- `dataset.py` / `inference.py` - data loading with eager `AuroraDataset` for training and lazy `LazyCMIP6Data` for inference.
  CMIP6 variable names are renamed to Aurora's ERA5-style names on load.
- `evaluate.py` - base vs fine-tuned vs persistence/climatology rollout
  metrics; results land in `eval_results/` for the notebook to plot.
- `demo.ipynb` - a demo.
- `slurm_scripts/` - SLURM batch scripts.
- `train.sh` / `train_chunk.sh` - shell scripts that call train.py via torchrun.
- `checkpoints/`, `runs/`, `eval_results/`, `slurm_logs/` - training and evaluation artifacts.

## Setup

Our scripts and notebooks expect to find two conda environments in the
root of the repository.
`mamba-env` is the environment for CPU-based training and `gpu-env` is for GPU-training.

On JASMIN, you should find that there are pre-existing environments in the shared workspace.
You can make symbolic links with:

```bash
ln -s /path/to/shared/workspace/dir/gpu-env gpu-env
ln -s /path/to/shared/workspace/dir/mamba-env mamba-env
```

### Creating new environments

If you do not have the pre-existing envs, we can build them from the specs in `envs/`.

Reminder: keep conda and pip from filling up your home directory by pointing them elsewhere:

```shell
export CONDA_PKGS_DIRS=/gws/ssde/j25b/turing_environment/users/$USER/conda-pkgs
export TMPDIR=/gws/ssde/j25b/turing_environment/users/$USER/tmp
export PIP_CACHE_DIR=$TMPDIR/pipcache
```

Reminder: If you want to create an environment without using the lock files in `envs/`, do so from the same kind of environment that you wish to run in.
In particular, build the environment for GPU training on an ORCID node.

#### mamba-env

This is used for notebooks, evaluation, tests, etc.

```shell
conda create -y -p ./mamba-env --file envs/mamba-env.explicit.txt

PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  conda env update -p ./mamba-env -f envs/mamba-env.export.yml

# make sure we use the newer, backwards-compatible libstdc++
conda env config vars set -p ./mamba-env LD_LIBRARY_PATH="$PWD/mamba-env/lib"

# register the notebook kernel for demo.ipynb
LD_LIBRARY_PATH=$PWD/mamba-env/lib \
  ./mamba-env/bin/python -m ipykernel install --user --name aurora-demo
```

#### gpu-env

Used for training.

```shell
conda create -y -p ./gpu-env --file envs/gpu-env.explicit.txt

# we need the Aurora fork for its chunk-and-check support
LD_LIBRARY_PATH=$PWD/gpu-env/lib ./gpu-env/bin/pip install chunkcheck==0.1.1
LD_LIBRARY_PATH=$PWD/gpu-env/lib ./gpu-env/bin/pip install --no-deps \
  -e /gws/ssde/j25b/turing_environment/users/$USER/aurora-hpc/aurora

conda env config vars set -p ./gpu-env LD_LIBRARY_PATH="$PWD/gpu-env/lib"
```

### Verifying

Check the conda layer matches the spec:

```shell
conda compare -p ./mamba-env envs/mamba-env.export.yml
conda compare -p ./gpu-env envs/gpu-env.export.yml
```

Then check the envs actually work:

```shell
# mamba-env: the unit tests exercise aurora + xarray + netcdf4 + dask
LD_LIBRARY_PATH=$PWD/mamba-env/lib ./mamba-env/bin/python \
  -m unittest discover -s tests -p 'test_*.py'

# gpu-env: the fork must import and resolve to the aurora submodule
LD_LIBRARY_PATH=$PWD/gpu-env/lib ./gpu-env/bin/python \
  -c "import aurora, chunkcheck; print(aurora.__file__)"
```

### Regenerating the specs after changing an env

```shell
conda env export -p <env-path> | grep -v '^prefix:' > envs/<env>.export.yml
conda list --explicit --md5 -p <env-path> > envs/<env>.explicit.txt
```

## Data

We expect there to be the following data in the root of the repo:

- `./data/0pt25_static.nc` the static data
- `./data/build/atmos.regridded` the regridded atmospheric variables
- `./data/build/surface.regridded` the regridded surface variables

As with the environments, some data in this format should already be available on the project file share.

## Running

- Training demo: `sbatch slurm_scripts/demo_train.sbatch` (1 GPU, 1 epoch
  ≈ 1 h; checkpoints to `checkpoints/demo/`, TensorBoard to
  `runs/demo-<jobid>`).
- Evaluation: `sbatch slurm_scripts/evaluate.sbatch` -> `eval_results/`.
- Tests: `LD_LIBRARY_PATH="$PWD/mamba-env/lib" ./mamba-env/bin/python -m
  unittest discover -s tests -p 'test_*.py'` from this directory.
- If the sci VM has no slurm client, submit via `ssh sci-vm-04` from this
  directory.
