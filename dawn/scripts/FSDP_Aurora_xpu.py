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
from load_batches import get_gt_batch, get_input_batch
from load_data import load_data
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

from aurora import Aurora

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
    # ddp_loss[1] += len(inp)

    print("Running all_reduce")
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, inp, gt):
    model.eval()
    device = torch.device(f"xpu:{rank}")
    correct = 0
    ddp_loss = torch.zeros(2).to(device)
    with torch.no_grad():
        inp = inp.to(device)
        gt = gt.to(device)
        output = model(inp)
        loss = mae(output, gt)
        ddp_loss[0] += loss.item()  # sum up batch loss
        # ddp_loss[1] += len(inp)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(
            "Test set loss: {:.4f}".format(
                ddp_loss[0],
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
    # test_input = get_input_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)

    # Get output
    train_gt = get_gt_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)
    # test_gt = get_gt_batch(1, static_vars_ds, surf_vars_ds, atmos_vars_ds)

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
    model = Aurora(use_lora=False, autocast=True).to(
        device
    )  # , timestep=timedelta(args.timestep)).to(device)
    # Specify the argument `device_ids` as XPU device in FSDP API.
    print(f"Wrapping FSDP model")
    model = FSDP(model, device_id=device)

    # AdamW, as used in the paper.
    optimizer = torch.optim.AdamW(model.parameters())

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model=model,
            rank=rank,
            world_size=world_size,
            inp=train_input,
            gt=train_gt,
            optimizer=optimizer,
            epoch=epoch,
        )
        # test(model, rank, world_size, test_input, test_gt)
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
