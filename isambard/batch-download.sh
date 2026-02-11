#!/bin/bash
# vim: et:ts=4:sts=4:sw=4
#SBATCH --time 0:10:0
#SBATCH --gpus 1
#SBATCH --output log-download.txt

# Launch with `sbatch batch-download.sh`.

echo
echo "## Aurora prepare script starting"

# Quit on error
set -e

export CDSAPI_RC="$HOME/.cdsapirc"

if [ ! -f "$CDSAPI_RC" ]; then
  echo "Please create a CDSAPI configuration file at $CDSAPI_RC. See https://github.com/ecmwf/cdsapi?tab=readme-ov-file#configure"
  exit 1
fi

. ../../isambard/environments/venv/bin/activate

echo
echo "## Downloading data"

python download.py ../../datasets/era5/

echo
echo "## Aurora prepare script completed"
