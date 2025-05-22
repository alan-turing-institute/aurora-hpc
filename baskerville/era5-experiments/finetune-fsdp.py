#!/usr/bin/env python
# vim: et:ts=4:sts=4:sw=4

# SPDX-License-Identifier: MIT
# Copyright 2025 The Alan Turing Institute

import os
import torch
import xarray as xr
import matplotlib.pyplot as plt
import torch.distributed as dist

from aurora import Aurora, rollout, Batch, Metadata
from pathlib import Path
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def print_memory_usage():
    print("Rank {} CUDA memory usage: {:.2f} GiB".format(os.environ["RANK"], torch.cuda.memory_allocated() / (1024**3)))

print("init")
print("World size: {}".format(os.environ['WORLD_SIZE']))
print("Rank: {}".format(os.environ["RANK"]))

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Fine-Tuning
# See https://microsoft.github.io/aurora/finetuning.html

print("loading model...")
model = Aurora(
    use_lora=False,  # Model was not fine-tuned.
    autocast=True,  # Use AMP.
)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
torch.cuda.set_device(local_rank)
model = model.to(local_rank)
model = FSDP(model)

# Data will be downloaded here.
download_path = Path("./downloads")
download_path = download_path.expanduser()

print("loading data...")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

i = 1  # Select this time index in the downloaded data.

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
model.configure_activation_checkpointing()

# AdamW, as used in the paper.
optimizer = torch.optim.AdamW(model.parameters())

# Not really necessary, for one forward pass.
optimizer.zero_grad()

print("performing forward pass...")
print_memory_usage()

with torch.autocast(device_type="cuda"):
    pred = model.forward(batch)
    #loss_fn = nn.CrossEntropyLoss()

    # space constraints
    batch = batch.to(local_rank)

    # mean absolute error of one variable
    print("calculating loss...")
    loss = torch.mean(torch.abs(pred.surf_vars["2t"] - batch.surf_vars["2t"][:,:,:720,:]))

    pred = pred.to("cpu")
    del batch

print("emptying cache...")
print_memory_usage()
torch.cuda.empty_cache()

print("performing backward pass...")
print_memory_usage()
loss.backward()

print("optimizing...")
optimizer.step()

print("deinit")
dist.destroy_process_group()

print("done")
