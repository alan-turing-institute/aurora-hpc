"""Fine tune Aurora weather model."""

print("importing...", flush=True)
import argparse
import os
import re
import time
import warnings
from pathlib import Path

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

from aurora import Aurora

parser = argparse.ArgumentParser()
parser.add_argument("--xpu", action="store_true", help="boolean of whether to use xpu")
parser.add_argument("--xpu-optimize", action="store_true", help="do ipex.optimize")
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
)
args = parser.parse_args()

if args.xpu:
    import intel_extension_for_pytorch as ipex

    # unset affinity mask
    os.environ.pop("ZE_AFFINITY_MASK", None)


def main(download_path: str, xpu: bool = False, xpu_optimize=False):
    if xpu:
        device_type = "xpu"
    else:
        comms_backend = "nccl"
        device_type = "cuda"

    time_start_total = time.time()

    device = f"{device_type}"
    print(f"Using {device=}", flush=True)

    print("loading model...", flush=True)
    model = Aurora(
        use_lora=False,  # Model was not fine-tuned.
        autocast=True,  # Use AMP.
    )
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    if not xpu:
        torch.cuda.set_device(LOCAL_RANK)
    else:
        torch.xpu.set_device("xpu:0")

    download_path = Path(download_path)

    print("preparing model...", flush=True)
    model.configure_activation_checkpointing()
    model = model.to(device)
    model.train()

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    if xpu and xpu_optimize:
        print("calling ipex.optimize...", flush=True)
        model, optimizer = ipex.optimize(model, optimizer=optimizer)


    print("loading data...", flush=True)
    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_filepath=Path("static.nc"),
        surface_filepath=Path("2023-01-surface-level.nc"),
        atmos_filepath=Path("2023-01-atmospheric.nc"),
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=False,  # We don't need to shuffle.
        collate_fn=aurora_collate_fn,
        num_workers=10,
        pin_memory=True,
    )

    times = []

    time_start = time.time()
    for batch, (X, y) in enumerate(data_loader):


        #X = X.to("xpu")
        optimizer.zero_grad()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            y = y.to(device)
            pred = model(X)

            # only one of these is necessary
            pred = pred.to(device)

            # mean absolute error of one variable
            print("calculating loss...", flush=True)

            # Todo: Are pred's of type PyTree and does it matter?
            loss = mae(pred, y)

        if batch > 4:
            break
        elif batch > 2:
            print("performing backward pass...", flush=True)
            starter = time.perf_counter()
            loss.backward()
            print("synchronizing")
            torch.xpu.synchronize()
            print("sync and backprop took", time.perf_counter() - starter)


        print(f"batch {batch}...", flush=True)

        time_end = time.time()
        times.append(time_end - time_start)
        print("batch took:", time_end - time_start, flush=True)
        time_start = time.time()

        time_end_total = time.time()
        print(f"Total time: {time_end_total - time_start_total}", flush=True)

    print("done", flush=True)


main(args.download_path, xpu=args.xpu, xpu_optimize=args.xpu_optimize)
