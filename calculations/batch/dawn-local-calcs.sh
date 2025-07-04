#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# ./dawn-local-calcs.sh

echo "## Aurora calculation testing script starting"

# Quit on error
#set -e

pushd ../

echo "## Loading modules"

module purge
module load default-dawn
module load lua
module load intel-oneapi-ccl/2021.14.0
module load intel-oneapi-mpi/2021.14.1
module load intel-oneapi-mkl/2025.0.1

echo "## Configuring environment"

# Merge tiles into full devices, for extra memory.
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# Avoid too many open file handles error.
ulimit -n 1000000

# Avoid mpi failing to init.
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=verbs

# Avoids segfaults, for some reason.
export ZES_ENABLE_SYSMAN=1

# Otherwise we're told to.
export CCL_ZE_IPC_EXCHANGE=sockets

echo "## Initialising virtual environment"

python3.12 -m venv venv
. ./venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
pip install --quiet intel-extension-for-pytorch==2.7.10+xpu oneccl_bind_pt==2.7.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install --quiet matplotlib

echo "## Running calculations"

# Perform the calculations
python calculate.py -a "xpu" -p "XPU " -o "calcs-dawn.csv"

echo "## Tidying up"

deactivate
popd

echo "## Aurora calculation testing script completed"
