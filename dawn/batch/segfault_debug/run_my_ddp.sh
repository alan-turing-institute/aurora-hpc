#!/usr/bin/env bash

# e.g.
# module load intel-oneapi-mpi/2021.14.1
# mpirun -n 2 -prepend-rank ./run_my_ddp.sh 

module purge
module load default-dawn
module load lua
module load intel-oneapi-ccl/2021.14.0
module load intel-oneapi-mpi/2021.14.1
module load intel-oneapi-mkl/2025.0.1

source ../../environments/venv_3_11_9/bin/activate

# These are now optional.
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets

# mpirun will try to set this for us.
unset ZE_AFFINITY_MASK

# See 2x tiles as one device.
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# OOM?
#export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Pass all arguments to allgather module.
python ../../scripts/segfault_debug/my_ddp.py

