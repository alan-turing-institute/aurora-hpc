"""Fine tune Aurora weather model."""

print("importing...")
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
from torch.profiler import ProfilerActivity, profile, record_function
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

    # get the master address
    numbers = re.compile("\d+")
    nodelist_env = os.getenv("SLURM_JOB_NODELIST")

    # e.g. "pvc-s-[24-25]"
    try:
        # If we're running on >1 node, we should set the MASTER_ADDR
        # to the hostname of rank 0.
        prefix = nodelist_env[0 : nodelist_env.index("[")]
        nodelist = tuple(prefix + x for x in numbers.findall(nodelist_env))
        master_addr = nodelist[0]
    except ValueError:
        # We must be running on a single node.
        master_addr = "0.0.0.0"

    os.environ["MASTER_ADDR"] = master_addr
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

    activities = [ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    if device_type == "xpu":
        activities.append(ProfilerActivity.XPU)
    print(f"Profiling activities: {activities}")

    time_start_total = time.time()

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

    print("loading data...")
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
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("train"):
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

    times = torch.Tensor(times).to(device)
    gathered_times = [torch.zeros(times.shape).to(device) for _ in range(WORLD_SIZE)]
    all_gather(gathered_times, times)

    if int(RANK) == 0:
        avg_time = sum([sum(t[1:]) for t in gathered_times]) / sum(
            [len(times[1:]) for t in gathered_times]
        )
        print(f"Average time per epoch (ignoring first): {avg_time} seconds")
        print(f"Effective time for an epoch: {avg_time / WORLD_SIZE} seconds")
        print(f"Equivalent training speed: {WORLD_SIZE / avg_time} epochs per seconds")
        total_time = sum([sum(t) for t in gathered_times])
        total_no_epochs = sum([len(t) for t in gathered_times])
        print(f"Total time for {total_no_epochs} epochs: {total_time}")

        time_end_total = time.time()
        print(f"Total time: {time_end_total - time_start_total}")

    destroy_process_group()
    print(
        f"Profiler results: \n{prof.key_averages().table(sort_by=f'{str(device)}_mem_usage', row_limit=10)}"
    )
    print("done")


main(args.download_path, xpu=args.xpu)
