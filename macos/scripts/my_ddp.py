"""Fine tune Aurora weather model."""

print("importing...")
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from aurora_loss import mae
from dataset import AuroraDataset, aurora_collate_fn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler

from aurora import Aurora

parser = argparse.ArgumentParser()
parser.add_argument("--xpu", action="store_true", help="boolean of whether to use xpu")
parser.add_argument(
    "--download_path",
    "-d",
    help="path to download directory",
    default="../../era5/era_v_inf",
)
args = parser.parse_args()

if args.xpu:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch  # has side-effects

    # unset affinity mask
    os.environ.pop("ZE_AFFINITY_MASK", None)

    # PMI_SIZE set by mpirun
    WORLD_SIZE = int(os.environ["PMI_SIZE"])
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)

    # PMI_RANK set by mpirun
    RANK = os.environ["PMI_RANK"]
    os.environ["RANK"] = RANK

    # MPI_LOCALRANKID provenance unknown
    LOCAL_RANK = int(os.environ["MPI_LOCALRANKID"])

    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "29876"
    USE_SUBDEVICES = os.environ.get("USE_SUBDEVICES", False)

else:
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])


def main(download_path: str, xpu: bool = False):
    if xpu:
        comms_backend = "ccl"
        device_type = "xpu"
    else:
        comms_backend = "nccl"
        device_type = "cuda"

    start_time_total = time.time()

    print("Initialising process group with backend", comms_backend, flush=True)
    # ToDo Run 2 or more processes.
    init_process_group(
        world_size=int(WORLD_SIZE),
        rank=int(RANK),
        backend=comms_backend,
    )

    device = f"{device_type}:{LOCAL_RANK}"
    print(f"Using {device=}")

    print("loading model...")
    model = Aurora(
        use_lora=False,  # Model was not fine-tuned.
        autocast=True,  # Use AMP.
    )
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    if not xpu:
        torch.cuda.set_device(LOCAL_RANK)

    download_path = Path(download_path)

    print("loading data...")

    # 1 for RANK 0 and 3 for RANK 1.
    i = (int(RANK) * 2) + 1
    print(f"batching with {i=}")

    print("preparing model...")
    model.configure_activation_checkpointing()
    model = FSDP(
        model,
        device_id=LOCAL_RANK,
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    model.train()

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    dataset = AuroraDataset(
        data_path=download_path,
        t=1,
        static_filepath=Path("static.nc"),
        surface_filepath=Path("2023-01-surface-level.nc"),
        atmos_filepath=Path("2023-01-atmospheric.nc"),
    )
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,  # If we set a batch size we'll need a collate_fn
        shuffle=False,  # We don't need to shuffle.
        sampler=sampler,
        collate_fn=aurora_collate_fn,
    )

    times = []

    time_start = time.time()
    for batch, (X, y) in enumerate(data_loader):
        print(f"batch {batch}...")

        optimizer.zero_grad()

        with torch.autocast(device_type=device_type):
            print("performing forward pass...")
            pred = model(X)

            # only one of these is necessary
            pred = pred.to(device)
            y = y.to(device)

            # mean absolute error of one variable
            print("calculating loss...")

            # Todo: Are pred's of type PyTree and does it matter?
            loss = mae(pred, y)

        print("performing backward pass...")
        loss.backward()

        print("optimizing...")
        optimizer.step()

        time_end = time.time()
        times.append(time_end - time_start)
        time_start = time.time()

    avg_time = sum(times[1:]) / len(times[1:])
    print(f"Average time per epoch (ignoring first): {avg_time}")
    print(f"Total time for {len(times)} epochs: {sum(times)}")

    end_time_total = time.time()
    print(f"Total time: {end_time_total - start_time_total}")

    destroy_process_group()
    print("done")


main(args.download_path, xpu=args.xpu)
