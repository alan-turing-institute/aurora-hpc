"""
Import Intel® extension for Pytorch\* and Intel® oneCCL Bindings for Pytorch\*
"""

import argparse
import functools
import os
from datetime import timedelta
from pathlib import Path

# Import Intel® extension for Pytorch\* and Intel® oneCCL Bindings for Pytorch\*
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from aurora_loss import mae
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    size_based_auto_wrap_policy,
    wrap,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from aurora import Aurora, Batch, Metadata


def load_data():
    """Load the ERA5 data"""
    download_path = Path("../era5/era_v_inf")

    print(f"Loading data..")
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(
        download_path / "2023-01-surface-level.nc", engine="netcdf4"
    )
    atmos_vars_ds = xr.open_dataset(
        download_path / "2023-01-atmospheric.nc", engine="netcdf4"
    )
    return static_vars_ds, surf_vars_ds, atmos_vars_ds


def get_input_batch(i: int, static_vars_ds, surf_vars_ds, atmos_vars_ds):
    """Get batch for i-th time index"""
    print(f"Getting batch for i={i} and i-1")
    batch = Batch(
        surf_vars={
            # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(
                int(level) for level in atmos_vars_ds.pressure_level.values
            ),
        ),
    )
    return batch


def get_gt_batch(i: int, static_vars_ds, surf_vars_ds, atmos_vars_ds):
    """Get i+1 data, i.e. prediction from i-th time index"""
    print(f"Getting ground truth results for i={i}")
    batch = Batch(
        surf_vars={
            # First select time point `i + 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i + 1]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i + 1]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i + 1]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i + 1]][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i + 1]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i + 1]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i + 1]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i + 1]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i + 1]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            # NOTE: Not sure what value should be here
            time=(
                surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i + 1],
            ),
            atmos_levels=tuple(
                int(level) for level in atmos_vars_ds.pressure_level.values
            ),
        ),
    )
    return batch


"""
Set the initialize the process group backend as Intel® oneCCL Bindings for Pytorch\*
"""


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
    os.environ["MASTER_PORT"] = "29500"  # your master port

    # initialize the process group by Intel® oneCCL Bindings for Pytorch\*
    dist.init_process_group("ccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, inp, gt, optimizer, epoch, sampler=None):
    model.train()
    device = torch.device(f"xpu:{rank}")
    ddp_loss = torch.zeros(2).to(device)
    if sampler:
        sampler.set_epoch(epoch)

    # single batch
    inp = inp.to(device)
    gt = gt.to(device)
    optimizer.zero_grad()
    output = model(inp)
    loss = mae(inp, gt)
    loss.backward()
    optimizer.step()
    ddp_loss[0] += loss.item()
<<<<<<< Updated upstream
    ddp_loss[1] += len(inp)
=======
    # ddp_loss[1] += len(inp)
>>>>>>> Stashed changes

    print("Running all_reduce")
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, inp, gt):
    model.eval()
    device = torch.device(f"xpu:{rank}")
    correct = 0
<<<<<<< Updated upstream
    ddp_loss = torch.zeros(3).to(device)
=======
    ddp_loss = torch.zeros(2).to(device)
>>>>>>> Stashed changes
    with torch.no_grad():
        inp = inp.to(device)
        gt = gt.to(device)
        output = model(inp)
        loss = mae(output, gt)
        ddp_loss[0] += loss.item()  # sum up batch loss
<<<<<<< Updated upstream
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
        ddp_loss[2] += len(inp)
=======
        # ddp_loss[1] += len(inp)
>>>>>>> Stashed changes

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
<<<<<<< Updated upstream
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
=======
        print(
            "Test set loss: {:.4f}".format(
                ddp_loss[0],
>>>>>>> Stashed changes
            )
        )


"""
Change the device related logic from 'rank' to 'f"xpu:{rank}"'.
Specify the argument `device_ids` as XPU device in FSDP API.
"""


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    # Load data
    static_vars_ds, surf_vars_ds, atmos_vars_ds = load_data()

    # Get input
    train_input = get_input_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)
<<<<<<< Updated upstream
    # test_input = get_batch(i=1)

    # Get output
    train_gt = get_gt_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)
    # test_gt = get_batch(i=5)
=======
    #test_input = get_input_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)

    # Get output
    train_gt = get_gt_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)
    #test_gt = get_gt_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)
>>>>>>> Stashed changes

    xpu_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    device = torch.device(f"xpu:{rank}")
    print(f"Setting device: {device}")
    torch.xpu.set_device(device)

    init_start_event = torch.xpu.Event(enable_timing=True)
    init_end_event = torch.xpu.Event(enable_timing=True)

    print("Setting up model")
<<<<<<< Updated upstream
    model = Aurora(use_lora=False, autocast=False).to(
        device
    )  # , timestep=timedelta(args.timestep)).to(device)
    # Specify the argument `device_ids` as XPU device in FSDP API.
    print(f"Moving FSDP model to {device}")
=======
    model = Aurora(use_lora=False, autocast=True).to(
        device
    )  # , timestep=timedelta(args.timestep)).to(device)
    # Specify the argument `device_ids` as XPU device in FSDP API.
    print(f"Wrapping FSDP model")
>>>>>>> Stashed changes
    model = FSDP(model, device_id=device)

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(
<<<<<<< Updated upstream
            args,
            model=model,
            rank=rank,
            world_size=world_size,
            inp=train_input,
            gt=train_gt,
            optimizer=optimizer,
            epoch=epoch,
        )
        # test(model, rank, world_size, test_batch)
=======
           args,
           model=model,
           rank=rank,
           world_size=world_size,
           inp=train_input,
           gt=train_gt,
           optimizer=optimizer,
           epoch=epoch,
        )
        #test(model, rank, world_size, test_input, test_gt)
>>>>>>> Stashed changes
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(
            f"XPU event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "aurora_ft.pt")

    cleanup()


"""
Replace CUDA runtime API with XPU runtime API.
"""
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Aurora FSDP example")
    parser.add_argument(
        "--timestep",
        type=int,
        default=6,
        help="timestep of the model (default: 6 (hours))",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # unset ZE_AFFINITY_MASK
    os.environ.pop("ZE_AFFINITY_MASK", None)

    WORLD_SIZE = torch.xpu.device_count()
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
