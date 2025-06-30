# Running Aurora on DAWN

## Set up (from scratch)

1. Create venv with python=3.11
2. Activate venv
3. Install aurora (`pip install microsoft-aurora`)
4. Install intel extension for pytorch as per [docs](https://pytorch-extension.intel.com/installation)

## Set up (quick)

1. Create venv with python=3.11
2. Activate venv
3. Install from requirements `pip install -r environments/requirements.txt`

# Whats in the repo?

- `scripts` - python scripts for running the Aurora code
- `batch` - sbatch scripts for running the python scripts
- `envrionments` - contains `requirements.txt` file, also where we are putting our python venv
