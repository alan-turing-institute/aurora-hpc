"""Fine tune Aurora weather model."""

from datetime import datetime
import time
print("importing...", flush=True)
import argparse
import os
import re
import warnings
from pathlib import Path

import warnings
import traceback

def custom_warn(message, category, filename, lineno, file=None, line=None):
    
    print(datetime.now().replace(microsecond=0), f"\n⚠️ {category.__name__}: {message}")
    traceback.print_stack(limit=5)

warnings.showwarning = custom_warn

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

import torch
import torch.nn as nn
from aurora_loss import mae
from dataset import AuroraDataset, aurora_collate_fn
from torch.distributed import all_gather, destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler

from aurora import Aurora, AuroraSmall
from functools import partial

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.xpu.utils import XPUComputeEng

# unset affinity mask
os.environ.pop("ZE_AFFINITY_MASK", None)

# Configurable parameters
M, K, N = 512, 512, 512  # Dimensions of the matrices
iterations = 100  # How many GEMMs to perform

def main():
    device_type = "xpu"

    device = f"{device_type}"
    print(datetime.now().replace(microsecond=0), f"Using {device=}", flush=True)

    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    C = torch.empty(M, N, device=device)

    # Warmup
    for _ in range(10):
        torch.matmul(A, B)

    time_start_total = time.perf_counter()
    for _ in range(100_000):
        C = torch.matmul(A, B)

    torch.xpu.synchronize()
    took = time.perf_counter() - time_start_total
    print(datetime.now().replace(microsecond=0), f"Took {took}", flush=True)


main()
