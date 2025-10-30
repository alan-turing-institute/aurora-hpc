#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# srun --qos turing --account usjs9456-ati-test --time 1:00:00 --nodes 1 --gpus 1 --cpus-per-gpu 36 --mem 0 --pty /bin/bash
# source ./batch-srun.sh

echo "## Aurora configuration script starting"

# Quit on error
# set -e

if [ ! -d ../era5-experiments/downloads ]; then
  echo "Please run the batch-download.sh script to download the data."
  exit 1
fi

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live

echo
echo "## Configuring environment"

echo "## Initialising virtual environment"

python3.11 -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet matplotlib
pip install --quiet -e ../../.[bask]

echo "## Aurora configuration script completed"
