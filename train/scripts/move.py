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
    print(datetime.now().replace(microsecond=0), f"Using {device=}", flush=True)

    AuroraI = partial(
            Aurora,
            encoder_depths=(2, 6, 2), 
            encoder_num_heads=(4, 8, 16),
            decoder_depths=(2, 6, 2),
            decoder_num_heads=(16, 8, 4),
            embed_dim=256,
            num_heads=8,
            use_lora=False,
            )
    #AuroraII =     #for constructor in [Aurora, AuroraSmall]: #, AuroraI, AuroraII]
    for i in range(5,30):
        print(datetime.now().replace(microsecond=0), "loading model...", i, flush=True)
        constructor = partial(
            Aurora,
            encoder_depths=(i, i, i),
            encoder_num_heads=(4, 8, 16),
            decoder_depths=(i, i, i),
            decoder_num_heads=(16, 8, 4),
            embed_dim=256,
            num_heads=8,
            )

        model = constructor(
            use_lora=False,  # Model was not fine-tuned.
            autocast=True,  # Use AMP.
        )
        #model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

        # Some sense of the size. See
        # https://discuss.pytorch.org/t/finding-model-size/130275
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(datetime.now().replace(microsecond=0), 'model size after: {:.3f}MB'.format(size_all_mb))

        if not xpu:
            torch.cuda.set_device(LOCAL_RANK)
        else:
            torch.xpu.set_device("xpu:0")

        download_path = Path(download_path)

        print(datetime.now().replace(microsecond=0), "preparing model...", flush=True)
        model.configure_activation_checkpointing()
        model = model.to(device)
        model.train()

        # AdamW, as used in the paper.
        optimizer = torch.optim.AdamW(model.parameters())

        if xpu and xpu_optimize:
            print(datetime.now().replace(microsecond=0), "calling ipex.optimize...", flush=True)
            model, optimizer = ipex.optimize(model, optimizer=optimizer)


        print(datetime.now().replace(microsecond=0), "loading data...", flush=True)
        dataset = AuroraDataset(
            data_path=download_path,
            t=1,
            static_filepath=Path("static.nc"),
            surface_filepath=Path("2023-01-01-surface-level.nc"),
            atmos_filepath=Path("2023-01-01-atmospheric.nc"),
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
                print(datetime.now().replace(microsecond=0), "calculating loss...", flush=True)

                # Todo: Are pred's of type PyTree and does it matter?
                loss = mae(pred, y)

            #if batch > 4:
            #elif batch > 2:
            print(datetime.now().replace(microsecond=0), "performing backward pass...", flush=True)
            starter = time.perf_counter()
            loss.backward()
            print(datetime.now().replace(microsecond=0), "synchronizing")
            torch.xpu.synchronize()
            print(datetime.now().replace(microsecond=0), "sync and backprop took", time.perf_counter() - starter)
            break


            print(datetime.now().replace(microsecond=0), f"batch {batch}...", flush=True)

            time_end = time.time()
            times.append(time_end - time_start)
            print(datetime.now().replace(microsecond=0), "batch took:", time_end - time_start, flush=True)
            time_start = time.time()

            time_end_total = time.time()
            print(datetime.now().replace(microsecond=0), f"Total time: {time_end_total - time_start_total}", flush=True)

    print(datetime.now().replace(microsecond=0), "done", flush=True)


main(args.download_path, xpu=args.xpu, xpu_optimize=args.xpu_optimize)
