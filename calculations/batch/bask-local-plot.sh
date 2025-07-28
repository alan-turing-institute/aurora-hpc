#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# ./bask-local-calcs.sh

echo "## Aurora calculation testing script starting"

# Quit on error
set -e

pushd ../

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load matplotlib/3.7.2-gfbf-2023a
module -q load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "## Configuring environment"

export OMP_NUM_THREADS=1

echo "## Initialising virtual environment"

python3 -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet typing_extensions==4.14.1

echo "## Plotting graphs"

# Render some graphs - GPU plots
python plot.py -x "Multiply-and-Add" -o "dev-mad-1024x1024-0042" -i "calcs-dawn-xpu-0042.csv" -i "calcs-bask-gpu-0042.csv"
python plot.py -x "Multiply-and-Add" -o "dev-mad-1024x1024-0043" -i "calcs-dawn-xpu-0043.csv" -i "calcs-bask-gpu-0043.csv"
python plot.py -x "Fuzzed Multiply-and-Add" -o "dev-mad-fuzz-1024x1024" -i "calcs-dawn-xpu-mad-fuzz-1024x1024.csv" -i "calcs-bask-gpu-mad-fuzz-1024x1024.csv"
python plot.py -x "Multiply-and-Add" -o "dev-mad-3x3" -i "calcs-dawn-xpu-mad-3x3.csv" -i "calcs-bask-gpu-mad-3x3.csv"
python plot.py -x "Multiply" -o "dev-mul-1024x1024" -i "calcs-dawn-xpu-mul-1024x1024.csv" -i "calcs-bask-gpu-mul-1024x1024.csv"
python plot.py -x "Addition" -o "dev-add-1024x1024" -i "calcs-dawn-xpu-add-1024x1024.csv" -i "calcs-bask-gpu-add-1024x1024.csv"

# Render some graphs - CPU plots
python plot.py -x "Multiply-and-Add" -o "cpu-mad-1024x1024-0042" -i "calcs-dawn-cpu-0042.csv" -i "calcs-bask-cpu-0042.csv"
python plot.py -x "Multiply-and-Add" -o "cpu-mad-1024x1024-0043" -i "calcs-dawn-cpu-0043.csv" -i "calcs-bask-cpu-0043.csv"
python plot.py -x "Fuzzed Multiply-and-Add" -o "cpu-mad-fuzz-1024x1024" -i "calcs-dawn-cpu-mad-fuzz-1024x1024.csv" -i "calcs-bask-cpu-mad-fuzz-1024x1024.csv"
python plot.py -x "Multiply-and-Add" -o "cpu-mad-3x3" -i "calcs-dawn-cpu-mad-3x3.csv" -i "calcs-bask-cpu-mad-3x3.csv"
python plot.py -x "Multiply" -o "cpu-mul-1024x1024" -i "calcs-dawn-cpu-mul-1024x1024.csv" -i "calcs-bask-cpu-mul-1024x1024.csv"
python plot.py -x "Addition" -o "cpu-add-1024x1024" -i "calcs-dawn-cpu-add-1024x1024.csv" -i "calcs-bask-cpu-add-1024x1024.csv"

echo "## Tidying up"

deactivate
popd

echo "## Aurora calculation testing script completed"
