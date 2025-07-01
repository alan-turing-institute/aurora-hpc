"""Fine tune Aurora weather model."""

print("importing...")
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from aurora_loss import mae
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader

from aurora import Aurora
from dataset import AuroraDataset, aurora_collate_fn

xpu = False

if xpu:
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
    comms_backend = "gloo"
    device_type = "xpu"
else:
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    comms_backend = "nccl"
    device_type = "cuda"

def main():
    print("Initialising process group with backend", "ccl", flush=True)
    start_time_total = time.time()

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

    download_path = Path("../../era5/era_v_inf")

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
        sharding_strategy=ShardingStrategy.NO_SHARD
    )
    model.train()

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    dataset = AuroraDataset(
            data_path=download_path,
            t=1,
            static_filepath=Path("static.nc"),
            surface_filepath=Path("2023-01-01-surface-level.nc"),
            atmos_filepath=Path("2023-01-01-atmospheric.nc"),
        )
    sampler = DistributedSampler(dataset) if False else None
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


main()
