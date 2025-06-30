#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import os
import torch
import xarray as xr
import matplotlib.pyplot as plt
import torch.distributed as dist

from aurora import Aurora, AuroraSmall, rollout, Batch, Metadata
from pathlib import Path
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

def print_memory_usage():
    print("Rank {} CUDA memory usage: {:.2f} GiB".format(os.environ["RANK"], torch.cuda.memory_allocated() / (1024**3)))

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

print("init")
print("World size: {}".format(world_size))
print("Rank: {}".format(rank))
print("Local rank: {}".format(local_rank))

dist.init_process_group("nccl", rank=rank, world_size=world_size)

def mae(x_hat_t, x_t):
    lamb = 2
    vs_va = 9
    surface = {
        "2t": 3.0,
        "msl": 1.5,
        "10u": 0.77,
        "10v": 0.66,
    }
    atmos = {
        "z": 2.8,
        "q": 0.78,
        "t": 1.7,
        "u": 0.87,
        "v": 0.6,
    }
    foo = sum(
        [
            (v / (720 * 1440)) * torch.sum(torch.abs(x_hat_t.surf_vars[k] - x_t.surf_vars[k][:,:,:720,:]))
            for k, v in surface.items()
        ]
    )
    bar = sum(
        [
            (v / (720 * 1440 * 13))
            * torch.sum(torch.abs(x_hat_t.atmos_vars[k] - x_t.atmos_vars[k][:,:,:,:720,:]))
            for k, v in atmos.items()
        ]
    )

    alpha = 0.25

    return (lamb / vs_va) * ((alpha * foo) + bar)

# Fine-Tuning
# See https://microsoft.github.io/aurora/finetuning.html

print("loading model...")
model = Aurora(
#model = AuroraSmall(
    use_lora=False,  # Model was not fine-tuned.
    autocast=True,  # Use AMP.
)
#model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
torch.cuda.set_device(local_rank)
model.configure_activation_checkpointing()
model.to(local_rank)
#model = DDP(
#    model,
#    device_ids=[local_rank],
#)
model = DDP(
    model,
).to(local_rank)

# Data will be downloaded here.
download_path = Path("./downloads")
download_path = download_path.expanduser()

print("loading data...")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

i = 1 + rank  # Select this time index in the downloaded data.

print("batching...")
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
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

print("preparing model...")
print_memory_usage()
model.train()

# AdamW, as used in the paper.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Not really necessary, for one forward pass.
optimizer.zero_grad()

print("performing forward pass...")
print_memory_usage()

with torch.autocast(device_type="cuda"):
    pred = model.forward(batch)

    # space constraints
    batch = batch.to(local_rank)

    # mean absolute error of one variable
    print("calculating loss...")
    loss = mae(pred, batch)

    #loss = loss.to(local_rank)
    #pred = pred.to(local_rank)
    #batch = batch.to(local_rank)
    #pred = pred.to("cpu")
    #batch = batch.to("cpu")

#print("emptying cache...")
#print_memory_usage()
#torch.cuda.empty_cache()

print("performing backward pass...")
print_memory_usage()
loss.backward()

print("optimizing...")
optimizer.step()

print("deinit")
dist.destroy_process_group()

print("done")
